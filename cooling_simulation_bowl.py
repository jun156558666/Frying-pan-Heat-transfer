"""
フライパン（ボウル形状）非定常熱伝導解析 — 1D シェルモデル
  シェル中立面: 底面（r:0→110mm）→コーナー（直線近似）→側壁（r=121mm, h=47mm）
  持ち手: 1D フィン方程式（根元=側壁上縁温度）
  加熱 t=0‥300s → 冷却 t=300‥600s
出力:
  bowl_top_view_300s.png
  bowl_top_view_600s.png
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized

plt.rcParams['font.family'] = 'Meiryo'

# ============================================================
# 0. パラメータ
# ============================================================
materials = {
    '銅':                  {'k': 390.0,  'rho': 7700, 'c': 390 },
    'アルミ (A5052P)':     {'k': 137.0,  'rho': 2860, 'c': 880 },
    '鉄（ねずみ鋳鉄）':    {'k':  45.0,  'rho': 7200, 'c': 510 },
    'ステンレス (SUS304)': {'k':  16.0,  'rho': 8000, 'c': 500 },
    'フェノール樹脂 (PF)': {'k': 0.2618, 'rho': 1400, 'c': 1900},
}
conditions = ['空焚き', '水入り']

T_amb_C   = 20.0;  T_amb   = T_amb_C   + 273.15
T_water_C = 90.0;  T_water = T_water_C + 273.15

# ---- 幾何 ----
q_s         = 15000.0
A_bottom_SW = 25611.2e-6
R_b         = np.sqrt(A_bottom_SW / np.pi)   # バーナー半径 ≈ 0.09028 m

L_b    = 0.110   # 底面外縁半径 [m]
H_b    = 0.003   # 底面板厚 [m]
t_s    = 0.002   # 側壁板厚 [m]
R_out  = 0.122   # 側壁外縁半径 [m]
H_side = 0.047   # 側壁高さ [m]

R_mid_side = R_out - t_s / 2      # 側壁中立面半径 = 0.121 m
y_bot_mid  = H_b / 2              # 底面中立面 y = 0.0015 m
y_wall_bot = H_b + t_s / 2        # 側壁中立面基端 y = 0.004 m

# 水入り: 水面高さ（SolidWorks 水冷側壁面積から推算）
A_water_SW  = 45522.56e-6
A_bot_inner = np.pi * L_b**2
H_w_side    = max(0.0, A_water_SW - A_bot_inner) / (2 * np.pi * R_mid_side)  # ≈ 0.00990 m

# ---- 対流係数 ----
h_air = 20.0;  h_wtr = 200.0

# ---- 持ち手（フィン） ----
L_h    = 0.180;  d_h = 0.025
P_fin  = np.pi * d_h
Ac_fin = np.pi * d_h**2 / 4
h_fin  = 20.0
Nf     = 104
dx_fin = L_h / (Nf - 1)
x_fin  = np.linspace(0, L_h, Nf)
fin_int = np.arange(1, Nf - 1)

# ---- 時間 ----
dt      = 0.1
Nt_heat = int(300.0 / dt)
Nt_cool = int(300.0 / dt)
Nt      = Nt_heat + Nt_cool

# ============================================================
# 1. シェル中立面ジオメトリ
# ============================================================
ds_target = 0.001  # 目標格子間隔 [m]

# 底面: r=0 → L_b, y=y_bot_mid (水平)
Ns_b = int(round(L_b / ds_target)) + 1   # 111
r_b  = np.linspace(0, L_b, Ns_b)
t_b  = np.full(Ns_b, H_b)
y_b  = np.full(Ns_b, y_bot_mid)

# コーナー: (L_b, y_bot_mid) → (R_mid_side, y_wall_bot) 直線近似
dr_c = R_mid_side - L_b              # 0.011 m
dy_c = y_wall_bot - y_bot_mid        # 0.0025 m
S_c  = np.sqrt(dr_c**2 + dy_c**2)   # ≈ 0.01128 m
Nc   = max(2, int(round(S_c / ds_target)))   # 11
p_c  = np.linspace(0, 1, Nc + 1)[1:]        # exclude junction at p=0
r_c  = L_b       + p_c * dr_c
y_c  = y_bot_mid + p_c * dy_c
t_c  = H_b       + p_c * (t_s - H_b)        # 3 mm → 2 mm 線形テーパー

# 側壁: r=R_mid_side (一定), y=y_wall_bot → y_wall_bot+H_side
Ns_w = int(round(H_side / ds_target)) + 1   # 48
r_w  = np.full(Ns_w, R_mid_side)
y_w  = np.linspace(y_wall_bot, y_wall_bot + H_side, Ns_w)
t_w  = np.full(Ns_w, t_s)

# 全セグメント連結（接続点を重複排除）
r_all = np.concatenate([r_b, r_c, r_w[1:]])
y_all = np.concatenate([y_b, y_c, y_w[1:]])
t_all = np.concatenate([t_b, t_c, t_w[1:]])
Ns    = len(r_all)

# 隣接ノード間の弧長
ds_arr = np.sqrt(np.diff(r_all)**2 + np.diff(y_all)**2)   # length Ns-1

# セグメント境界
idx_c_start = Ns_b          # コーナー先頭インデックス (=111)
idx_w_start = Ns_b + Nc     # 側壁先頭インデックス (=122)

print(f"シェルノード数: 底面{Ns_b} + コーナー{Nc} + 側壁{Ns_w-1} = {Ns}")
print(f"R_b={R_b*1000:.1f}mm  H_w_side={H_w_side*1000:.1f}mm\n")

# ============================================================
# 2. シェル Backward Euler ビルダー
#    エネルギー式(円筒座標):
#      ρc t dT/dt = (1/r) d/ds(k t r dT/ds) + h_o(T_o-T) + h_i(T_i-T) + qs
#    i=0: L'Hôpital → 4k t / ds²
#    i=Ns-1: 断熱 Neumann
# ============================================================
def build_shell(k, rho, c, cond):
    # ---- ノードごとの BC アレイ ----
    h_o_arr = np.full(Ns, h_air)   # 外面（バーナー側・外壁側）は常に空気
    T_o_arr = np.full(Ns, T_amb)
    h_i_arr = np.zeros(Ns)
    T_i_arr = np.full(Ns, T_amb)
    qs_arr  = np.where(r_all <= R_b, q_s, 0.0)  # バーナー熱流束

    for i in range(Ns):
        if i < idx_w_start:
            # 底面 & コーナー: 内面は調理面（水入りは全面水冷）
            if cond == '水入り':
                h_i_arr[i] = h_wtr;  T_i_arr[i] = T_water
            else:
                h_i_arr[i] = h_air
        else:
            # 側壁: 内面は高さに応じて水冷
            if cond == '水入り' and y_all[i] <= H_b + H_w_side:
                h_i_arr[i] = h_wtr;  T_i_arr[i] = T_water
            else:
                h_i_arr[i] = h_air

    h_conv   = h_o_arr + h_i_arr
    coeff_arr = rho * c * t_all / dt   # ρ c t / dt  (shape Ns)

    A       = lil_matrix((Ns, Ns))
    b0_heat = np.zeros(Ns)
    b0_cool = np.zeros(Ns)

    for i in range(Ns):
        h_tot = h_conv[i]
        T_force = h_o_arr[i]*T_o_arr[i] + h_i_arr[i]*T_i_arr[i]

        if i == Ns - 1:
            # 上縁: 断熱 Neumann (A T[-1] = A T[-2])
            A[i, i]   = 1.0
            A[i, i-1] = -1.0
            # b stays 0

        elif i == 0:
            # 中心軸 r=0: L'Hôpital limit → 4k t / ds_p²
            ds_p = ds_arr[0]
            keff = 4.0 * k * t_all[0] / ds_p**2
            A[0, 0] = coeff_arr[0] + keff + h_tot
            A[0, 1] = -keff
            b0_heat[0] = T_force + qs_arr[0]
            b0_cool[0] = T_force

        else:
            ds_m   = ds_arr[i-1]
            ds_p   = ds_arr[i]
            ds_avg = 0.5 * (ds_m + ds_p)

            r_mh = 0.5 * (r_all[i-1] + r_all[i])
            r_ph = 0.5 * (r_all[i]   + r_all[i+1])
            t_mh = 0.5 * (t_all[i-1] + t_all[i])
            t_ph = 0.5 * (t_all[i]   + t_all[i+1])

            c_m = k * t_mh * r_mh / (r_all[i] * ds_m * ds_avg)
            c_p = k * t_ph * r_ph / (r_all[i] * ds_p * ds_avg)

            A[i, i]   = coeff_arr[i] + c_m + c_p + h_tot
            A[i, i-1] = -c_m
            A[i, i+1] = -c_p
            b0_heat[i] = T_force + qs_arr[i]
            b0_cool[i] = T_force

    return factorized(A.tocsr()), b0_heat, b0_cool, coeff_arr, qs_arr

# ============================================================
# 3. 持ち手 1D Backward Euler ビルダー
# ============================================================
def build_fin(k, rho, c):
    coeff_f = rho * c / dt
    hpac_f  = h_fin * P_fin / Ac_fin
    A = lil_matrix((Nf, Nf))
    for i in range(Nf):
        if i == 0:
            A[i, i] = 1.0
        elif i == Nf - 1:
            A[i, i] = 1.0;  A[i, i-1] = -1.0
        else:
            A[i, i]   = coeff_f + hpac_f + 2*k/dx_fin**2
            A[i, i-1] = -k/dx_fin**2
            A[i, i+1] = -k/dx_fin**2
    return factorized(A.tocsr()), coeff_f, hpac_f

# ============================================================
# 4. 非定常計算 (加熱 300s → 冷却 300s)
#    戻り値: {300: (T_sh_C, T_fin_C, T_max_all, T_min_tip),
#             600: (...)}
#    T_max_all: 外面（バーナー側）補正後の最高温度
#    T_min_tip: 持ち手先端温度
# ============================================================
def run_transient(k, rho, c, cond):
    solve_sh, b0h, b0c, coeff_sh, qs_arr = build_shell(k, rho, c, cond)
    solve_f,  coeff_f, hpac_f             = build_fin(k, rho, c)

    T_sh = np.full(Ns, T_amb)
    T_f  = np.full(Nf, T_amb)
    snap = {}

    # シェルの PDE ノード (Neumann BC ノードを除く)
    sh_int = np.arange(0, Ns - 1)

    for step in range(Nt):
        heating = (step < Nt_heat)

        # シェル更新
        b_sh = (b0h if heating else b0c).copy()
        b_sh[sh_int] += coeff_sh[sh_int] * T_sh[sh_int]
        T_sh = solve_sh(b_sh)

        # 持ち手: 根元 = 側壁上縁
        T_root = float(T_sh[-1])
        b_f = np.zeros(Nf)
        b_f[0]      = T_root
        b_f[fin_int] = coeff_f * T_f[fin_int] + hpac_f * T_amb
        T_f = solve_f(b_f)

        if step == Nt_heat - 1:
            T_sh_C = T_sh - 273.15
            T_f_C  = T_f  - 273.15
            # 外底面（バーナー側）の補正: T_outer = T_mid + q_s*t/(2k)
            corr = qs_arr * t_all / (2.0 * k)   # 0 for non-heated nodes
            T_outer_C = T_sh_C + corr
            T_max = float(max(T_outer_C.max(), T_f_C.max()))
            T_min = float(T_f_C[-1])
            snap[300] = (T_sh_C.copy(), T_f_C.copy(), T_max, T_min)

    T_sh_C = T_sh - 273.15
    T_f_C  = T_f  - 273.15
    T_max  = float(max(T_sh_C.max(), T_f_C.max()))
    T_min  = float(T_f_C[-1])
    snap[600] = (T_sh_C.copy(), T_f_C.copy(), T_max, T_min)
    return snap

# ============================================================
# 5. 全条件計算
# ============================================================
results = {}
for cond in conditions:
    for name, mat in materials.items():
        k, rho, c = mat['k'], mat['rho'], mat['c']
        print(f"  {name} / {cond} ...", end=' ', flush=True)
        snap = run_transient(k, rho, c, cond)
        results[(cond, name)] = snap
        for t_key, (T_sh_C, T_f_C, T_max, T_min) in snap.items():
            print(f"t={t_key}s: 最高{T_max:.1f}°C 最低{T_min:.1f}°C", end='  ')
        print()
    print()

# ============================================================
# 6. 可視化共通設定
# ============================================================
Nx_c, Ny_c = 900, 450
x_min_mm = -R_out * 1000 * 1.08
x_max_mm = (R_out + L_h) * 1000 * 1.06
y_lim_mm =  R_out * 1000 * 1.08

x_comp = np.linspace(x_min_mm, x_max_mm, Nx_c)
y_comp = np.linspace(-y_lim_mm, y_lim_mm, Ny_c)
X_comp, Y_comp = np.meshgrid(x_comp, y_comp)
R_comp     = np.sqrt(X_comp**2 + Y_comp**2) / 1000   # [m]
x_fin_comp = X_comp / 1000 - R_out                   # [m]

pan_bottom  = R_comp <= L_b
pan_rim     = (R_comp > L_b) & (R_comp <= R_out)
handle_mask = ((x_fin_comp >= 0) & (x_fin_comp <= L_h)
               & (np.abs(Y_comp / 1000) <= d_h / 2))
pan_bottom  = pan_bottom & ~handle_mask
pan_rim     = pan_rim    & ~handle_mask

EXTENT_TV = [x_min_mm, x_max_mm, -y_lim_mm, y_lim_mm]
BASE      = r'C:\Users\jun1029\claude code\大学\３年\固体力学'
PHASES    = {300: '加熱終了（t=300s）', 600: '冷却終了（t=600s）'}

r_bot = r_all[:Ns_b]   # 底面中立面ノードの r 座標

# ============================================================
# 7. 上面視図
# ============================================================
def plot_top_view(t_snap):
    fig, axes = plt.subplots(2, 5, figsize=(28, 11))
    plt.subplots_adjust(hspace=0.50, wspace=0.30)

    for row, cond in enumerate(conditions):
        for col, name in enumerate(materials):
            T_sh_C, T_f_C, T_max, T_min = results[(cond, name)][t_snap]
            T_tip = float(T_f_C[-1])
            vmin  = min(T_sh_C.min(), T_tip)
            vmax  = T_max

            T_comp = np.full((Ny_c, Nx_c), np.nan)

            # 底面: 中立面温度（調理面に近似）
            T_comp[pan_bottom] = np.interp(R_comp[pan_bottom], r_bot, T_sh_C[:Ns_b])

            # リム部（コーナー+側壁）: シェル上の r=L_b 付近の温度で一様着色
            T_rim_val = float(T_sh_C[Ns_b])
            T_comp[pan_rim] = T_rim_val

            # 持ち手
            T_comp[handle_mask] = np.interp(x_fin_comp[handle_mask], x_fin, T_f_C)

            ax = axes[row, col]
            im = ax.imshow(T_comp, extent=EXTENT_TV, origin='lower',
                           cmap='jet', aspect='equal', vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.72, aspect=18)
            cbar.set_label('°C', fontsize=7)
            cbar.ax.tick_params(labelsize=6)

            ax.add_patch(plt.Circle((0, 0), R_b*1000,  fill=False, color='white',
                                    ls='--', lw=1.2))
            ax.add_patch(plt.Circle((0, 0), L_b*1000,  fill=False, color='white',
                                    ls='-',  lw=0.8, alpha=0.7))
            ax.add_patch(plt.Circle((0, 0), R_out*1000, fill=False, color='white',
                                    ls='-',  lw=0.8, alpha=0.4))
            hw = d_h / 2 * 1000
            for sgn in [1, -1]:
                ax.plot([R_out*1000, (R_out+L_h)*1000], [sgn*hw, sgn*hw],
                        'w-', lw=0.8, alpha=0.6)

            ax.set_xlim(x_min_mm, x_max_mm);  ax.set_ylim(-y_lim_mm, y_lim_mm)
            ax.set_xlabel('x [mm]', fontsize=7);  ax.set_ylabel('y [mm]', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_title(f'{name} | {cond}\n最高 {T_max:.1f}°C  最低 {T_tip:.1f}°C',
                         fontsize=8.5)

    plt.suptitle(
        f'フライパン（ボウル形状 1D シェル）上面視  {PHASES[t_snap]}\n'
        f'q_s={q_s:.0f} W/m²  R_b={R_b*1000:.1f}mm  '
        f'底面r={L_b*1000:.0f}mm  側壁h={H_side*1000:.0f}mm',
        fontsize=12
    )
    out = fr'{BASE}\bowl_top_view_{t_snap}s.png'
    plt.savefig(out, dpi=150, bbox_inches='tight');  plt.close()
    print(f"保存: {out}")

# ============================================================
# 8. 出力
# ============================================================
for t_snap in [300, 600]:
    plot_top_view(t_snap)
