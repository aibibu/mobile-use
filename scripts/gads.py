"""Read a GADS iPhone screen, or explicitly run an AI task on it."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr

# This example does not send SDK telemetry. Model requests require --goal.
os.environ["MOBILE_USE_TELEMETRY_ENABLED"] = "false"

from minitap.mobile_use.clients.appium_client import RemoteAppiumClientWrapper
from minitap.mobile_use.clients.remote_appium_config import RemoteAppiumClientConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", help="Explicitly run this natural-language task")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum LangGraph execution steps, not the number of phone actions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Save an observation, or per-round task evidence when used with --goal",
    )
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_steps > 100:
        parser.error("--max-steps must be between 1 and 100")
    return args


def connection_config() -> RemoteAppiumClientConfig:
    # Resolve the documented local file independently of the caller's cwd.
    # Explicit process variables always take precedence over the file.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    required = ("GADS_APPIUM_URL", "GADS_DEVICE_UDID", "GADS_CLIENT_SECRET")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError("Missing environment variables: " + ", ".join(missing))
    return RemoteAppiumClientConfig(
        hub_url=os.environ["GADS_APPIUM_URL"],
        udid=os.environ["GADS_DEVICE_UDID"],
        client_secret=SecretStr(os.environ["GADS_CLIENT_SECRET"]),
    )


async def observe(config: RemoteAppiumClientConfig, output_dir: Path | None) -> None:
    client = RemoteAppiumClientWrapper(config=config)
    try:
        await client.init_client()
        width, height = await client.get_screen_size()
        screenshot = await client.screenshot()
        elements = await client.describe_all()
        summary: dict[str, int | str] = {
            "width": width,
            "height": height,
            "element_count": len(elements),
        }
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            for name, data in (
                ("screen.png", screenshot),
                ("elements.json", json.dumps(elements, ensure_ascii=False, indent=2).encode()),
            ):
                path = output_dir / name
                with path.open("wb") as stream:
                    os.chmod(path, 0o600)
                    stream.write(data)
            summary["output_dir"] = str(output_dir)
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        await client.cleanup()


async def run(args: argparse.Namespace) -> None:
    config = connection_config()
    if not args.goal:
        await observe(config, args.output_dir)
        return

    from minitap.mobile_use.sdk import Agent
    from minitap.mobile_use.config import initialize_llm_config
    from minitap.mobile_use.sdk.builders.agent_config_builder import AgentConfigBuilder
    from minitap.mobile_use.sdk.types.task import AgentProfile

    # The generic builder's implicit profile uses defaults. Make this CLI's
    # local override file explicit, without loading model credentials in observe mode.
    agent_config = (
        AgentConfigBuilder()
        .add_profile(AgentProfile(name="gads", llm_config=initialize_llm_config()))
        .build()
    )
    agent_config.remote_appium_config = config
    agent = Agent(config=agent_config)
    try:
        await agent.init()
        request = (
            agent.new_task(args.goal)
            .with_max_steps(args.max_steps)
            .with_trace_recording(
                args.output_dir is not None,
                str(args.output_dir) if args.output_dir is not None else None,
            )
            .build()
        )
        result = await agent.run_task(request=request)
        if result is not None:
            print(result)
    finally:
        await agent.clean()


if __name__ == "__main__":
    try:
        asyncio.run(run(arguments()))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception:
        # WebDriver errors can contain server payloads; do not print credentials or a tree.
        raise SystemExit(
            "GADS operation failed; check configuration and device availability."
        ) from None
