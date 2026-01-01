#!/usr/bin/env python3
import re
import sys
from pathlib import Path

pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)s/it")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"用法: {Path(sys.argv[0]).name} <日志文件路径>", file=sys.stderr)
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if not log_path.is_file():
        print(f"未找到日志文件: {log_path}", file=sys.stderr)
        sys.exit(1)

    times = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                times.append(float(match.group(1)))

    if not times:
        print("未找到任何 s/it 时间片段", file=sys.stderr)
        sys.exit(1)

    avg = sum(times) / len(times)
    print(f"共匹配 {len(times)} 条，平均用时: {avg:.2f}s/it")


if __name__ == "__main__":
    main()

