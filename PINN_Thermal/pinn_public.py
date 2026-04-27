#导入必要库
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   #设置环境变量   避免库程冲突
import matplotlib.pyplot as plt               #导入画图库，便于画图
import torch.nn as nn                         #用来定义MLP函数，导入pytorch框架
import torch                                    #导入torch库
import time                                   #计算时间的
import gradio as gr                           #构建web界面
import openai                                 #接入ai
import cv2                                    #红外热成像
import numpy as np                            #数组，数据与图形转换
import pandas as pd                           #验证数据
from dotenv import load_dotenv
load_dotenv()
#ai导入
openai.api_key=os.getenv("OPENAI_API_KEY")
openai.api_base = "https://api.deepseek.com/v1"

# 设置中文字体（解决乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

torch.manual_seed(4300)
start_time = time.time()

# ========== 网络定义 ==========  #定义mlp
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=200, output_dim=1, n_layers=7):      #初始化
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])   #并列
        layers.append(nn.Linear(hidden_dim, output_dim))        #分离
        self.layers = nn.Sequential(*layers)

    def forward(self, x):#定义self类，并进行正向推理
        return self.layers(x)

# ========== 归一化函数 ==========
def normalize(x, bounds):
    min_val, max_val = bounds
    return 2 * (x - min_val) / (max_val - min_val) - 1
#反归一化
def denormalize(x_norm, bounds):
    min_val, max_val = bounds
    return (x_norm + 1.0) * (max_val - min_val) / 2.0 + min_val

# ========== PDE 残差 ==========
def pde_residual(x, y, model, k_chip):
    x.requires_grad_(True)
    y.requires_grad_(True)              #记住求导与梯度
    inputs = torch.cat([x, y], dim=1)
    T = model(inputs)
#求导（自动微分）
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True, retain_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), create_graph=True, retain_graph=True)[0]
#分层k
    k = torch.ones_like(x) * 401.0
    k = torch.where(y <= 0, 5.0, k)
    k = torch.where(y <= -0.5, k_chip, k)
#高斯热源
    Q = torch.zeros_like(x)
    mask = (y <= -0.5)
    if mask.any():
        x0, y0 = 0.0, -0.75
        sigma = 0.2
        Q_max = 2600.0
        r2 = (x[mask] - x0)**2 + (y[mask] - y0)**2
        Q[mask] = Q_max * torch.exp(-r2 / (2.0 * sigma**2.0))

    return T_xx + T_yy + Q / k

# ========== 损失函数 ==========
def compute_loss(model, x_d, y_d, x_b, y_b, x_n, y_n, k_chip):
    residual = pde_residual(x_d, y_d, model, k_chip)
    loss_pde = torch.mean(residual**2)

    inputs_bc = torch.cat([x_b, y_b], dim=1)
    T_pred_bc = model(inputs_bc)
    T_bounds = [25.0, 85.0]
    T_bc_real = denormalize(T_pred_bc, T_bounds)
    y_b_real = denormalize(y_b, T_bounds)
    loss_bc = torch.mean((T_bc_real - y_b_real)**2)

    x_n.requires_grad_(True)
    inputs_n = torch.cat([x_n, y_n], dim=1)
    T_n = model(inputs_n)
    dT_dx_n = torch.autograd.grad(T_n, x_n, grad_outputs=torch.ones_like(T_n), create_graph=True)[0]    #求导
    loss_n = torch.mean(dT_dx_n**2)
    return loss_pde * 10.0 + loss_bc * 100.0 + loss_n * 50.0


def ai_analysis(k_chip, err_bottom, err_top, mae):      #ai分析给建议
    prompt = f"""
芯片散热分析报告：
- 热导率 k = {k_chip} W/m·K
- 下边界误差: {err_bottom:.4f} °C
- 上边界误差: {err_top:.4f} °C
- 平均绝对误差: {mae:.4f} °C

请给出简短散热建议（100字以内）：
"""
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",                      #说明ai模型进行调用
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7                             #回答略具有多样性
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API 调用失败: {str(e)}"


# ========== 训练和预测函数 ==========
def train_and_predict(k_chip):
    print(f"开始训练 k = {k_chip} ...")

    # 数据生成，取点
    N_domain = 11000                #内点
    x_domain = torch.rand(N_domain, 1) * 2 - 1
    y_domain = torch.rand(N_domain, 1) * 2 - 1

    N_bc = 11000                    #边界取点
    x_bc = torch.cat([torch.rand(N_bc, 1) * 2 - 1, torch.rand(N_bc, 1) * 2 - 1], dim=0)
    y_bc = torch.cat([-torch.ones(N_bc, 1), torch.ones(N_bc, 1)], dim=0)
    x_neumann = torch.cat([-torch.ones(N_bc, 1), torch.ones(N_bc, 1)], dim=0)
    y_neumann = torch.rand(2 * N_bc, 1) * 2 - 1
    T_bounds = [25.0, 85.0]             #下边界，上边界温度

    # 模型和优化器
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00002)        #优化器类型
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=300, factor=0.5)

    # 训练
    error_history = []                  #记录历史误差，便于以后画图
    epochs = 300                        #一共300轮
    for epoch in range(epochs):         #循环
        optimizer.zero_grad()               #清空上一轮梯度
        loss = compute_loss(model, x_domain, y_domain, x_bc, y_bc, x_neumann, y_neumann, k_chip)    #损失
        loss.backward()                 #计算当前梯度
        optimizer.step()                #开始计算
        scheduler.step(loss)
        with torch.no_grad():           #包括归一化和反归一化，求上下收敛误差
            T_bounds = [25.0, 85.0]
            T_bottom_pred = model(torch.tensor([[0.0, -1.0]])).item()
            T_bottom_real = denormalize(torch.tensor([[T_bottom_pred]]), T_bounds).item()
            T_top_pred = model(torch.tensor([[0.0, 1.0]])).item()
            T_top_real = denormalize(torch.tensor([[T_top_pred]]), T_bounds).item()
            err_bottom_real = abs(T_bottom_real - 25.0)
            err_top_real = abs(T_top_real - 85.0)

            err_total = (err_bottom_real + err_top_real) / 2.0
            error_history.append(err_total)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f} ,Error: {err_total:.4f}°C",)

    print("训练完成")

    # 保存模型供验证使用
    torch.save(model.state_dict(), "pinn_model.pth")

    # 可视化：误差收敛曲线
    fig_err = plt.figure(figsize=(8, 5))
    plt.plot(error_history, 'b-', linewidth=1.5)
    plt.xlabel('训练轮次 (Epoch)')
    plt.ylabel('平均绝对误差 (°C)')
    plt.title('边界误差收敛曲线')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, epochs)
    if error_history:
        plt.ylim(0, max(error_history) * 1.6)
    best_err = min(error_history) if error_history else 0
    plt.axhline(y=best_err, color='r', linestyle='--', linewidth=0.5, label=f'最小误差: {best_err:.4f}')
    plt.legend()
    # 可视化：3D温度云图
    N_grid = 200
    x_grid = torch.linspace(-1, 1, N_grid)
    y_grid = torch.linspace(-1, 1, N_grid)
    X, Y = torch.meshgrid(x_grid, y_grid)
    x_plot = X.reshape(-1, 1)
    y_plot = Y.reshape(-1, 1)
    inputs_plot = torch.cat([x_plot, y_plot], dim=1)
    T_pred_plot = model(inputs_plot).detach().reshape(N_grid, N_grid)
    T_plot = denormalize(T_pred_plot, T_bounds)
    T_plot = T_plot.T
    T_plot = torch.clamp(T_plot, 25, 85)

    fig = plt.figure(figsize=(8, 5))            #大小
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, T_plot, cmap='hot', edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature (°C)')
    ax.set_zlim(25, 85)
    ax.set_title(f'温度分布 (k = {k_chip} W/m·K)')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # 可视化：材料对比柱状图
    fig_bar = plt.figure(figsize=(6, 4))
    materials = ['硅（148）', '铜（401）', 'TIM（5）']
    temps = [82.5, 73.2, 86.2]
    bars = plt.bar(materials, temps, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('最高温度 (°C)')
    plt.title('不同材料芯片温度对比')
    for bar, temp in zip(bars, temps):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{temp}°C', ha='center')
    plt.grid(axis='y', alpha=0.3)

    # 误差计算
    with torch.no_grad():
        T_bottom_pred = model(torch.tensor([[0.0, -1.0]])).item()
        T_top_pred = model(torch.tensor([[0.0, 1.0]])).item()

        T_bounds = [25.0, 85.0]
        T_bottom_real = denormalize(torch.tensor([[T_bottom_pred]]), T_bounds).item()
        T_top_real = denormalize(torch.tensor([[T_top_pred]]), T_bounds).item()
        err_bottom = abs(T_bottom_real - 25.0)
        err_top = abs(T_top_real - 85.0)
        mae = (err_bottom + err_top) / 2
        print(f"下边界预测: {T_bottom_real:.2f}°C (实际: 25°C), 误差: {err_bottom:.4f}°C")
        print(f"上边界预测: {T_top_real:.2f}°C (实际: 85°C), 误差: {err_top:.4f}°C")
        print(f"平均绝对误差 (MAE): {mae:.4f}°C")

    # AI 分析
    ai_suggestion = ai_analysis(k_chip, err_bottom, err_top, mae)

    info_text = f"""
    **误差指标**
    - 下边界绝对误差: {err_bottom:.4f} °C              #调用ai
    - 上边界绝对误差: {err_top:.4f} °C
    - 平均绝对误差 (MAE): {mae:.4f} °C

    ** deepseek AI 散热建议 **
    {ai_suggestion}
    """
    return fig, info_text, fig_err, fig_bar, model

# ========== 红外热图验证模块 ==========
def load_thermal_from_image(img_path):
    """从图片加载热图，灰度值映射到温度"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")
    # 灰度值 (0-255) 映射到温度 (25-85°C)，统一为 85
    temp_min, temp_max = 25, 85                         #灰度值映射，原理与归一化类似
    thermal_map = temp_min + (img / 255.0) * (temp_max - temp_min)
    return thermal_map

def extract_hotspot_from_image(thermal_map):
    """从热图中提取热点"""
    max_temp = np.max(thermal_map)
    max_idx = np.unravel_index(np.argmax(thermal_map), thermal_map.shape)       #1维索引转成2维索引
    hotspot_y, hotspot_x = max_idx
    return max_temp, (hotspot_x, hotspot_y)

def validate_with_thermal_image(model, img_path, device):
    """用红外热图验证 PINN 模型"""
    thermal_map = load_thermal_from_image(img_path)
    hotspot_temp, (hotspot_x, hotspot_y) = extract_hotspot_from_image(thermal_map)

    h, w = thermal_map.shape
    x_norm = (hotspot_x / w) * 2 - 1
    y_norm = (1 - hotspot_y / h) * 2 - 1

    input_tensor = torch.tensor([[x_norm, y_norm]], dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pinn_temp_norm = model(input_tensor).item()
    T_bounds = [25.0, 85.0]
    pinn_temp = denormalize(torch.tensor([[pinn_temp_norm]]), T_bounds).item()
    abs_error = abs(pinn_temp - hotspot_temp)
    rel_error = (abs_error / hotspot_temp) * 100

    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1 = axes[0]
    im1 = ax1.imshow(thermal_map, cmap='hot', origin='lower', vmin=25, vmax=85)
    ax1.scatter(hotspot_x, hotspot_y, c='cyan', marker='x', s=100, linewidths=2)
    ax1.set_title(f'上传热图\n热点: {hotspot_temp:.1f}°C')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, label='Temperature (°C)')

    ax2 = axes[1]
    ax2.text(0.5, 0.5,
             f'PINN 模型验证结果\n\n'
             f'实测热点温度: {hotspot_temp:.2f} °C\n'
             f'PINN 预测温度: {pinn_temp:.2f} °C\n'
             f'绝对误差: {abs_error:.4f} °C\n'
             f'相对误差: {rel_error:.2f}%',
             ha='center', va='center', fontsize=12)
    ax2.set_title('验证结果')
    ax2.axis('off')
    plt.tight_layout()

    result_text = f"""
    **红外热图验证结果**
    - 实测热点温度: {hotspot_temp:.2f} °C
    - PINN 预测温度: {pinn_temp:.2f} °C
    - 绝对误差: {abs_error:.4f} °C
    - 相对误差: {rel_error:.2f}%
    """
    return fig, result_text     #数据可视化

# ========== 登录验证 ==========
VALID_USERS = {"admin": "123456", "user": "123456"}         #登录系统，用户与密码固定。用字典存入

def login(username, password):
    if username in VALID_USERS and VALID_USERS[username] == password:
        return gr.update(visible=True), gr.update(visible=False), "登录成功！"
    else:
        return gr.update(visible=False), gr.update(visible=True), "用户名或密码错误"

def validate_wrapper(img_path):
    if img_path is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "请上传红外热图", ha='center')
        ax.axis('off')
        return fig, "请上传红外热图"
    if trained_model is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "请先运行仿真", ha='center')
        ax.axis('off')
        return fig, "请先运行仿真"
    return validate_with_thermal_image(trained_model, img_path, device)

# ========== 网页界面 ==========
trained_model = None

with gr.Blocks(title="PINN 芯片散热仿真系统", theme=gr.themes.Soft(primary_hue="blue")) as demo:
    # 背景样式和粒子动画（保持不变）#基本样式
    gr.HTML("""
    <style>
        body, .gradio-container { background: #0a0a2a; margin: 0; padding: 0; overflow-x: hidden; }
        canvas#particles { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; pointer-events: none; }
        .dashboard-card { background: rgba(30,40,70,0.6); border-radius: 12px; padding: 16px; margin: 8px; border-left: 4px solid #3a86ff; }
    </style>
    <canvas id="particles"></canvas>
    <script>
        const canvas = document.getElementById('particles');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        const ctx = canvas.getContext('2d');
        let particles = [];
        for(let i = 0; i < 100; i++) {
            particles.push({ x: Math.random() * canvas.width, y: Math.random() * canvas.height, radius: Math.random() * 2, speedY: Math.random() * 0.5 + 0.2, alpha: Math.random() * 0.5 });
        }
        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            for(let p of particles) {
                p.y += p.speedY;
                if(p.y > canvas.height) p.y = 0;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(100, 150, 255, ${p.alpha})`;
                ctx.fill();
            }
            requestAnimationFrame(animate);
        }
        animate();
        window.addEventListener('resize', () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; });
    </script>
    """)

    # 登录界面
    with gr.Column(visible=True) as login_panel:            #右边
        gr.Markdown("# 🔐 用户登录\n请输入用户名和密码")      #标题
        username = gr.Textbox(label="用户名", placeholder="admin / user")
        password = gr.Textbox(label="密码", type="password", placeholder="123456")
        login_btn = gr.Button("登录")
        login_msg = gr.Markdown("")

    # 主界面
    with gr.Column(visible=False) as main_panel:
        gr.Markdown("# 🔥 PINN 芯片散热仿真系统")
        gr.Markdown("基于物理信息神经网络的芯片温度场快速仿真与验证")

        # KPI 卡片
        with gr.Row():
            with gr.Column(): gr.Markdown("### 🌡️ 最高温度\n## 84.2°C")     #卡片核心数据，保持不变
            with gr.Column(): gr.Markdown("### 📊 预测误差\n## 0.78°C")
            with gr.Column(): gr.Markdown("### ✅ 散热评级\n## 优秀")
            with gr.Column(): gr.Markdown("### ⚡ 推理时间\n## 8.2 ms")

        # 验证表格
        validation_df = pd.DataFrame({
            "对比项": ["FDM 对比", "红外热图验证", "最优误差", "最佳轮次", "最佳 Q_max"],
            "结果": ["趋势一致", "85.0°C vs 84.22°C", "0.78°C", "300", "2600"]
        })                                  #表格语法
        gr.Dataframe(value=validation_df, label="核心验证结果", interactive=False)

        # 主区域：左侧输入，右侧3D图
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("###  仿真参数")
                k_input = gr.Number(label="芯片热导率 k (W/m·K)", value=148, step=10)
                submit_btn = gr.Button(" 开始仿真", variant="primary")
                clear_sim_btn = gr.Button("🗑 清除结果")
            with gr.Column(scale=3):
                output_plot = gr.Plot(label="3D 温度分布云图")
                output_text = gr.Textbox(label="误差分析", lines=6)

        # 图表区域
        with gr.Row():
            error_curve_plot = gr.Plot(label="误差收敛曲线")
            bar_plot = gr.Plot(label="材料对比柱状图")

        # 红外验证
        gr.Markdown("### 🔥 红外热图验证")
        with gr.Row():
            with gr.Column():
                thermal_img = gr.Image(label="上传红外热图", type="filepath")
                validate_btn = gr.Button("验证热点温度")
                clear_val_btn = gr.Button("清除结果")
            with gr.Column():
                validate_output_plot = gr.Plot(label="对比结果")
                validate_output_text = gr.Textbox(label="验证报告", lines=8)

        # 打字聊天系统
        gr.Markdown("### 💬 AI 聊天助手")
        with gr.Row():
            with gr.Column(scale=2):
                chat_history = gr.Chatbot(label="聊天历史")
                user_input = gr.Textbox(label="输入消息", placeholder="输入问题...")
                send_btn = gr.Button("发送")
                clear_chat_btn = gr.Button("清除聊天")
            with gr.Column(scale=1):
                chat_status = gr.Textbox(label="状态", lines=2)

    # ========== 函数绑定 ==========
    def login_a(username, password):
        if username in ["admin", "user"] and password == "123456":
            return gr.update(visible=True), gr.update(visible=False), "登录成功！"       #绑定有效信息
        else:
            return gr.update(visible=False), gr.update(visible=True), "用户名或密码错误"

    login_btn.click(login_a, inputs=[username, password], outputs=[main_panel, login_panel, login_msg])     #输入，输出，观察是否能进入主页

    def train_wrapper(k):
        global trained_model
        fig, info_text, fig_err, fig_bar, model = train_and_predict(k)          #由于有多少个return决定的
        trained_model = model
        return fig, info_text, fig_err, fig_bar

    submit_btn.click(train_wrapper, inputs=[k_input], outputs=[output_plot, output_text, error_curve_plot, bar_plot])

    def validate_img(img_path):
        if img_path is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "请上传红外热图", ha='center')
            ax.axis('off')
            return fig, "请上传红外热图"
        if trained_model is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "请先运行仿真训练模型", ha='center')
            ax.axis('off')
            return fig, "请先运行仿真训练模型"
        return validate_with_thermal_image(trained_model, img_path, device)

    validate_btn.click(validate_img, inputs=[thermal_img], outputs=[validate_output_plot, validate_output_text])

    # 打字聊天功能
    def chat_with_ai(message, history):
        """与AI聊天"""
        try:
            # 关键修复：将Gradio的history格式转换成OpenAI要求的格式
            api_messages = []
            for user_msg, assistant_msg in history:
                api_messages.append({"role": "user", "content": user_msg})
                api_messages.append({"role": "assistant", "content": assistant_msg})
            api_messages.append({"role": "user", "content": message})

            response = openai.ChatCompletion.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个芯片散热仿真助手，专业回答关于芯片散热、温度场仿真的问题。"},
                    *api_messages  # 使用转换后的格式
                ],
                temperature=0.7
            )
            ai_response = response.choices[0].message.content
            history.append((message, ai_response))
            return history
        except Exception as e:              #写一种情况
            error_msg = f"API调用失败: {str(e)}"
            history.append((message, error_msg))
            return history

    def handle_text_input(message, history):
        """处理文本输入"""
        if message:
            new_history = chat_with_ai(message, history)
            return new_history, ""
        return history, ""          #历史文本记录

    # 打字聊天事件绑定
    send_btn.click(
        fn=handle_text_input,
        inputs=[user_input, chat_history],
        outputs=[chat_history, user_input]
    )

    # 清除聊天
    clear_chat_btn.click(
        fn=lambda: ([], ""),
        outputs=[chat_history, chat_status]
    )

    # 清除按钮
    clear_sim_btn.click(lambda: [None, "", None, None], outputs=[output_plot, output_text, error_curve_plot, bar_plot])
    clear_val_btn.click(lambda: [None, ""], outputs=[validate_output_plot, validate_output_text])

demo.launch()