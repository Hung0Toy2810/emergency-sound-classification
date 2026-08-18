"""
adaptive_controller.py — Gradient-Informed Per-Layer Capacity Growth
=====================================================================
Tối ưu tham số ở CẤP ĐỘ TỪNG PHẦN TỬ TUYẾN TÍNH (Conv2d layer).

Cơ chế:
  1. Đo ||∇W|| (gradient norm) cho TỪNG Conv2d sau mỗi epoch
  2. Layer nào có gradient norm cao nhất → đang "chịu áp lực" lớn nhất
     → cần thêm capacity (thêm channels) để giảm loss
  3. Chỉ tăng channels cho stage chứa layer đó
  4. Nhân tử Lagrange kiểm soát: dừng tăng khi overfit

Toán học:
  Gradient norm cho layer l:
    g_l = ||∂L/∂W_l||_F   (Frobenius norm)

  Đo "áp lực" cho mỗi stage s (mỗi stage có 2 conv layers):
    pressure_s = g_{conv1_s} + g_{conv2_s}

  Quy tắc tăng:
    s* = argmax_s pressure_s     (stage chịu áp lực lớn nhất)
    width_s* += Δw               (chỉ tăng stage đó)

  Ràng buộc Lagrange:
    λ = max(0, λ + α*(val_loss - train_loss - ε))
    Nếu λ ≥ threshold → DỪNG tăng (overfit)

  Ý nghĩa: thay vì tăng đều tất cả (lãng phí), ta "bơm" tham số
  chính xác vào nơi mà gradient báo hiệu là đang thiếu capacity.
"""

import torch
import torch.nn as nn
import copy
import numpy as np

BASE_CHANNELS = [32, 64, 128, 256, 512]
NUM_STAGES = 5


class LagrangeCapacityController:
    """
    Controller với per-stage width multipliers, grow theo gradient pressure.
    Mỗi stage (pair of conv layers) có width_mult riêng.

    Cơ chế chống bùng nổ tham số — dựa trên Parameter Efficiency Score (PES):

      PES = ΔAcc / (ΔParams / Params_trước_grow)
      = (accuracy tăng) / (tỷ lệ params tăng)

      Ví dụ:
        +100% params, +0.5% acc  → PES = 0.005/1.0 = 0.005  → rất tệ → PHẠT NẶNG
        +5%   params, +3%   acc  → PES = 0.03/0.05 = 0.600  → tốt     → không phạt

      Mức phạt (theo ngưỡng PES):
        PES < min_pes              → phạt nẶNG: bad_grows += 2 → khóa ngay
        min_pes ≤ PES < good_pes   → phạt nhẹ: bad_grows += 1
        PES ≥ good_pes             → grow hiệu quả: reset bad_grows = 0

      Các cơ chế hỗ trợ thêm:
        - Cooldown: chờ grow_cooldown epochs giữa các lần grow
        - Lagrange: dừng grow khi overfit (gap quá lớn)
        - (Tùy chọn) Hard budget: max_total_params nếu muốn giới hạn cứng
    """
    def __init__(self, min_width=0.15, delta_width=0.05, max_width=1.0,
                 epsilon=0.08, alpha=0.5, lambda_threshold=1.0,
                 plateau_patience=5, max_total_params=None,
                 grow_cooldown=8,
                 min_pes=0.02, good_pes=0.10, min_abs_acc=0.003):
        # Per-stage width multipliers
        self.stage_widths = [min_width] * NUM_STAGES
        self.min_width = min_width
        self.max_width = max_width
        self.delta_width = delta_width
        self.epsilon = epsilon
        self.alpha = alpha
        self.lambda_threshold = lambda_threshold
        self.plateau_patience = plateau_patience

        # --- Anti-explosion controls (PES-based) ---
        self.max_total_params = max_total_params      # None = không giới hạn cứng
        self.grow_cooldown = grow_cooldown             # Epochs chờ giữa các grow
        self.min_pes = min_pes                         # PES < min_pes → phạt NẶNG (block ngay)
        self.good_pes = good_pes                       # PES ≥ good_pes → hiệu quả (nếu acc cũng đủ)
        self.min_abs_acc = min_abs_acc                 # ΔAcc tuyệt đối tối thiểu để grow được coi là tốt

        self.epochs_since_last_grow = 999              # Lớn → có thể grow ngay từ đầu
        self.acc_at_last_grow = 0.0                    # Accuracy khi grow lần cuối
        self.params_at_last_grow = 0                   # Params NGAY TRƯỚC grow (trước khi thêm)
        self.pes_checked = True                        # Đã kiểm tra PES của grow hiện tại chưa
        self.consecutive_ineffective_grows = 0         # Số grows kém liên tiếp
        self.last_pes = None                           # PES của grow vừa rồi (dữ liệu log)

        self.lam = 0.0
        self.best_val_loss = float('inf')
        self.no_improve_count = 0
        self.history = []
        self.converged = False
        self.grow_count = 0
        self.gradient_pressures = [0.0] * NUM_STAGES
        self.current_val_acc = 0.0

    @property
    def width_mult(self):
        """Width trung bình (để hiển thị)."""
        return sum(self.stage_widths) / NUM_STAGES

    def _estimate_params(self, channels=None):
        """Ước tính tổng tham số (chỉ Conv2d + BN + Linear) từ channels."""
        if channels is None:
            channels = self.get_channels()
        c = channels
        total = 0
        # 5 blocks: mỗi block có conv1(c_in→c_out) + conv2(c_out→c_out) + 2 BN
        in_channels = [3] + c[:4]
        for i in range(5):
            total += in_channels[i] * c[i] * 9      # conv1 (3x3)
            total += c[i] * c[i] * 9                # conv2 (3x3)
            total += c[i] * 4                        # 2 BN layers (weight + bias each)
        # Classifier: Linear(c[4]→128) + Linear(128→num_classes≈10)
        total += c[4] * 128 + 128 + 128 * 10 + 10
        return total

    def step(self, train_loss, val_loss, epoch, val_acc=0.0):
        gap = val_loss - train_loss
        self.lam = max(0.0, self.lam + self.alpha * (gap - self.epsilon))
        self.current_val_acc = val_acc
        self.epochs_since_last_grow += 1

        # --- PES check: đánh giá grow vừa rồi có hiệu quả không ---
        # Kiểm tra một lần DUY NHẤT tại giữa cooldown
        if (not self.pes_checked and
                self.epochs_since_last_grow == self.grow_cooldown // 2):
            self.pes_checked = True

            # Tính ΔAcc và ΔParams
            delta_acc = val_acc - self.acc_at_last_grow
            params_now = self._estimate_params()
            params_ratio = (params_now - self.params_at_last_grow) / (self.params_at_last_grow + 1e-8)

            # PES = ΔAcc / ΔParams_ratio
            pes = delta_acc / (params_ratio + 1e-8)
            self.last_pes = pes

            # Điều kiện "grow tốt" cần THỎA MÃN CẢ HAI:
            #   1. PES ≥ good_pes  (hiệu suất tương đối tốt)
            #   2. delta_acc ≥ min_abs_acc  (cải thiện tuyệt đối đủ lớn)
            # Ngăn edge case: params tăng ít (→ PES cao) nhưng acc hầu như không đổi
            acc_ok = delta_acc >= self.min_abs_acc

            if pes < self.min_pes or (not acc_ok and pes < self.good_pes):
                # Cực kỳ lãng phí: tham số tăng nhiều nhưng acc không đáng kể
                # → Phạt nặng ngay: +2 điểm → block luôn
                self.consecutive_ineffective_grows += 2
                print(f"  ⚠ PES={pes:.4f} < {self.min_pes} → PHẠT NẶNG "
                      f"(params +{params_ratio*100:.1f}%, acc +{delta_acc*100:.2f}%)")
            elif not acc_ok or pes < self.good_pes:
                # Kém hiệu quả: PES tốt nhưng acc tuyệt đối không đủ, hoặc PES trung bình
                self.consecutive_ineffective_grows += 1
                print(f"  ⚠ PES={pes:.4f} (trung bình) → phạt nhẹ "
                      f"(params +{params_ratio*100:.1f}%, acc +{delta_acc*100:.2f}%)")
            else:
                # Grow hiệu quả: PES tốt VÀ acc tăng đủ lớn → thưởng: reset điểm phạt
                self.consecutive_ineffective_grows = max(0, self.consecutive_ineffective_grows - 1)
                print(f"  ✓ PES={pes:.4f} ≥ {self.good_pes}, Δacc={delta_acc*100:.2f}% → grow tốt "
                      f"(params +{params_ratio*100:.1f}%)")

        if val_loss < self.best_val_loss - 0.001:
            self.best_val_loss = val_loss
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        is_plateau = self.no_improve_count >= self.plateau_patience
        can_grow = any(w < self.max_width - 1e-6 for w in self.stage_widths)
        not_overfitting = self.lam < self.lambda_threshold
        cooldown_ok = self.epochs_since_last_grow >= self.grow_cooldown

        # Hard budget check (chỉ khi user đặt max_total_params)
        under_budget = True
        if self.max_total_params is not None:
            under_budget = self._estimate_params() < self.max_total_params

        # Consecutive ineffective grows → block further growth
        not_wasting = self.consecutive_ineffective_grows < 2

        should_grow = (is_plateau and can_grow and not_overfitting
                       and cooldown_ok and under_budget and not_wasting
                       and not self.converged)

        if is_plateau and not not_overfitting:
            self.converged = True

        # If we're over budget and plateaued, mark converged
        if self.max_total_params is not None and is_plateau and not under_budget:
            self.converged = True

        info = {
            'lambda': self.lam, 'gap': gap, 'epsilon': self.epsilon,
            'width_avg': self.width_mult, 'converged': self.converged,
            'stage_widths': list(self.stage_widths),
            'pressures': list(self.gradient_pressures),
            'est_params': self._estimate_params(),
            'budget': self.max_total_params or 'unlimited',
            'cooldown_remaining': max(0, self.grow_cooldown - self.epochs_since_last_grow),
            'bad_grows': self.consecutive_ineffective_grows,
            'last_pes': self.last_pes,
        }
        self.history.append(info)
        return should_grow, info

    def update_pressures(self, pressures):
        """Cập nhật gradient pressure cho từng stage."""
        self.gradient_pressures = list(pressures)

    def grow_targeted(self):
        """Tăng width cho stage có gradient pressure cao nhất.

        Trước khi grow: lưu lại params_at_last_grow và acc_at_last_grow
        để sau cooldown//2 epochs có thể tính PES chính xác.
        """
        # Chỉ xét stages chưa đạt max
        candidates = [(p, i) for i, (p, w) in
                       enumerate(zip(self.gradient_pressures, self.stage_widths))
                       if w < self.max_width - 1e-6]
        if not candidates:
            return -1

        # Adaptive delta: giảm bước grow theo mức độ phạt tích lũy
        # 0 bad grows → delta 100%
        # 1 bad grow  → delta 50%
        # 2+ bad grows→ không được grow (bị chặn ở should_grow)
        if self.consecutive_ineffective_grows >= 1:
            effective_delta = self.delta_width * 0.5
        else:
            effective_delta = self.delta_width

        # Chọn stage có pressure cao nhất
        _, target_stage = max(candidates, key=lambda x: x[0])

        # Ghi lại params TRƯỚC khi grow (cho PES)
        self.params_at_last_grow = self._estimate_params()

        # Thực hiện grow
        new_width = min(self.max_width, self.stage_widths[target_stage] + effective_delta)
        old_widths = list(self.stage_widths)
        self.stage_widths[target_stage] = new_width

        if self.max_total_params is not None and self._estimate_params() > self.max_total_params:
            # Revert — vượt budget
            self.stage_widths = old_widths
            self.converged = True
            return -1

        self.no_improve_count = 0
        # Không reset best_val_loss = inf — chỉ nới lỏng 5% để model mới có cơ hội
        self.best_val_loss *= 1.05

        # Reset tracking cho PES của grow lần này
        self.acc_at_last_grow = self.current_val_acc
        self.pes_checked = False   # Chưa kiểm tra PES — sẽ check sau grow_cooldown//2 epochs
        self.last_pes = None
        self.epochs_since_last_grow = 0
        self.grow_count += 1
        return target_stage

    def get_channels(self):
        """Tính channels thực tế cho từng stage."""
        return [max(2, round(BASE_CHANNELS[i] * self.stage_widths[i]))
                for i in range(NUM_STAGES)]


# ============================================================================
# Gradient Pressure Measurement
# ============================================================================

def compute_gradient_pressures(model):
    """
    Đo gradient norm cho từng Conv2d layer, tổng hợp theo stage.

    Mỗi stage có 2 Conv2d (conv1, conv2).
    pressure_s = ||∇W_{conv1}||_F + ||∇W_{conv2}||_F

    Ý nghĩa: stage nào có gradient lớn → loss nhạy cảm nhất
    với trọng số ở đó → cần thêm capacity.
    """
    pressures = [0.0] * NUM_STAGES
    blocks = _get_blocks(model)

    for i, block in enumerate(blocks):
        block_pressure = 0.0
        for name, param in block.named_parameters():
            if 'conv' in name and 'weight' in name and param.grad is not None:
                # Relative gradient: |∇W| / (|W| + ε)
                # Scale-invariant: đo "loss muốn thay đổi weight bao nhiêu so với giá trị hiện tại"
                # Công bằng giữa block to và block nhỏ
                rel_grad = (param.grad.data.abs() / (param.data.abs() + 1e-8)).mean().item()
                block_pressure += rel_grad
        pressures[i] = block_pressure

    # Normalize để so sánh công bằng giữa các stages
    total = sum(pressures) + 1e-10
    pressures = [p / total for p in pressures]
    return pressures


def _get_blocks(model):
    """Lấy 5 conv blocks từ model (hỗ trợ cả GrowableCNN và DKL)."""
    if hasattr(model, 'block1'):
        return [model.block1, model.block2, model.block3,
                model.block4, model.block5]
    elif hasattr(model, 'feature_extractor'):
        fe = model.feature_extractor
        return [fe.block1, fe.block2, fe.block3, fe.block4, fe.block5]
    raise ValueError("Unknown model structure")


# ============================================================================
# Growable Conv Block + Model
# ============================================================================

class GrowableConvBlock(nn.Module):
    """Conv2d→BN→ReLU→Conv2d→BN→ReLU→MaxPool2d"""
    def __init__(self, c_in, c_out, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        return self.pool(x)


class GrowableCNN(nn.Module):
    """CNN với channels per-stage độc lập."""
    def __init__(self, num_classes, channels=None, dropout_conv=0.2, dropout_fc=0.4):
        super().__init__()
        if channels is None:
            channels = BASE_CHANNELS
        c = channels
        self.block1 = GrowableConvBlock(3, c[0])
        self.block2 = GrowableConvBlock(c[0], c[1])
        self.block3 = GrowableConvBlock(c[1], c[2], dropout=dropout_conv)
        self.block4 = GrowableConvBlock(c[2], c[3], dropout=dropout_conv)
        self.block5 = GrowableConvBlock(c[3], c[4], dropout=dropout_conv)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_fc),
            nn.Linear(c[4], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )
        self.channels = list(channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.classifier(x)


def grow_model(old_model, new_channels, num_classes, device):
    """
    Net2Net style: tạo model mới, copy trọng số kênh cũ, Kaiming init kênh mới.
    """
    new_model = GrowableCNN(num_classes, channels=new_channels).to(device)
    old_sd = old_model.state_dict()
    new_sd = new_model.state_dict()
    for key in old_sd:
        old_t = old_sd[key]
        new_t = new_sd[key]
        if old_t.shape == new_t.shape:
            new_sd[key] = old_t
        else:
            slices = tuple(slice(0, min(o, n)) for o, n in
                          zip(old_t.shape, new_t.shape))
            new_sd[key][slices] = old_t[slices]
    new_model.load_state_dict(new_sd)
    return new_model


def setup_cosine_lr(optimizer, T_0=8, T_mult=2, eta_min=1e-6):
    """Cosine Annealing with Warm Restarts."""
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )
