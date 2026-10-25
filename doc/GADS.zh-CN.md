# 通过 GADS 连接远程 iPhone

这个 fork 基于 `minitap-ai/mobile-use` 的提交
`88daffb20281e0f0a4bb5c5049005c078f63cad6`，增加 GADS 远程 Appium 连接。
复用原有的 Agent、iOS 控制器和 Appium 操作；本机 Android/iOS、BrowserStack
以及其他云设备连接仍保留。

```text
mobile-use Agent
  → RemoteAppiumClientWrapper
  → GADS Hub /grid
  → Provider 上的 Appium / XCUITest
  → iPhone 上的 WDA
```

首版支持 iOS。Android 仍使用上游已有连接方式，尚未接入这个远程入口。
这里没有新增 GADS API，也不直接连接或公开 WDA 端口。

## 前置条件

1. iPhone 已在 GADS 中在线，并允许自动化使用。
2. 对应 Provider 已启用 `Setup Appium servers`，Appium 和 XCUITest 驱动可用。
3. 已有可访问目标设备的 GADS Appium client secret。
4. 使用 Hub 的 `/grid` 地址，不能使用已停用的 `/device/{id}/appium`。

查看 `/grid/status` 可以检查 Grid 是否就绪。带
`Authorization: Bearer <client secret>` 的请求才能查看凭证可访问的设备；
`ready=true` 不代表目标手机当前空闲。

GADS 官方说明：[Grid](https://github.com/shamanec/GADS/blob/v6.1.0/docs/hub.md#appium-grid)、
[Provider](https://github.com/shamanec/GADS/blob/v6.1.0/docs/provider.md#appium---optional)。

## 安装和配置

```bash
uv sync --frozen --extra dev

export GADS_APPIUM_URL='http://127.0.0.1:18100/grid'
export GADS_DEVICE_UDID='<目标 iPhone 的 UDID>'
export GADS_CLIENT_SECRET='<已有的 GADS Appium client secret>'
```

本机地址只适用于已经建立的 GADS 转发。也可以配置可访问的 HTTPS Hub 地址。
凭证放在进程环境或本地未提交的配置中；不要写进脚本或 Git。

远程连接使用明确的设备 ID，并设置 `noReset=true`、`fullReset=false`、
`autoLaunch=false`。不会提交应用安装包。Appium 的这些选项不能代替真机验收：
首次使用前仍需确认目标设备的登录态和应用状态没有被 Provider 配置改变。

## 先验证读屏，再运行 AI

默认示例只创建会话、读取截图与元素数量，然后释放会话。它不调用模型，也不点击：

```bash
uv run --frozen python scripts/gads.py
```

需要保存本地观察结果时，显式指定输出目录：

```bash
uv run --frozen python scripts/gads.py --output-dir artifacts/gads/observe
```

截图和元素可能包含页面中的私密内容。该目录已忽略，不会随代码提交。

配置好上游支持的模型后，再显式传入自然语言任务：

```bash
uv run --frozen python scripts/gads.py \
  --goal '读取当前测试页面的标题和按钮文字，不点击、不提交' \
  --max-steps 10
```

模型配置沿用上游 `.env`、`llm-config.override.jsonc` 和 SDK profiles。
示例关闭遥测；只有传入 `--goal` 才进入 Agent 执行流程。

## Python SDK

```python
import os

from pydantic import SecretStr

from minitap.mobile_use.clients.remote_appium_config import RemoteAppiumClientConfig
from minitap.mobile_use.sdk import Agent
from minitap.mobile_use.sdk.builders.agent_config_builder import AgentConfigBuilder

config = AgentConfigBuilder().build()
config.remote_appium_config = RemoteAppiumClientConfig(
    hub_url=os.environ["GADS_APPIUM_URL"],
    udid=os.environ["GADS_DEVICE_UDID"],
    client_secret=SecretStr(os.environ["GADS_CLIENT_SECRET"]),
)
agent = Agent(config=config)
try:
    await agent.init()
    await agent.run_task(goal="读取当前测试页面上的文字，不提交任何内容")
finally:
    await agent.clean()
```

截图、元素位置、点击使用同一逻辑坐标系；错误应回传给 Agent，不能当成点击成功。
程序正常结束、初始化失败或取消后，都应释放自己创建的远程会话。

## 验证范围

自动化测试使用替代 WebDriver 验证连接参数、截图和坐标、失败传播、会话清理，
以及 SDK 选择远程客户端而不启动本机设备服务。
它们不证明已经连通某台真机，也不证明业务流程通过。

真机验收需另外完成：同一会话读屏 → 找到测试控件 → 点击/输入 → 观察结果 →
释放会话，并确认登录态保持。HTTP、JS 调用栈和 HAR 采集是后续独立步骤。
