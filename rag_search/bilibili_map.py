"""纪录片名到B站BVID的映射及工具函数。"""

from typing import Dict

# 纪录片名（解析后的 doc_name） → BVID 映射
_DOC_BVID_MAP: Dict[str, str] = {
    "BBC Light Fantastic 2004 01": "BV1xx411c7mD",
}

# demo 占位 BVID
_PLACEHOLDER_BVID = "BV1xx411c7mD"


def parse_doc_name(title: str) -> str:
    """从文件名解析纪录片展示名。

    "BBC.Light.Fantastic.2004.01.srt" → "BBC Light Fantastic 2004 01"
    """
    name = title.strip()
    if name.lower().endswith(".srt"):
        name = name[:-4]
    return name.replace(".", " ")


def parse_start_seconds(time_str: str) -> float:
    """从时间范围字符串解析起始秒数。

    "00:01:25,318 --> 00:02:23,039" → 85.318
    """
    if not time_str:
        return 0.0
    # 取起始部分
    start = time_str.split(" --> ")[0] if " --> " in time_str else time_str
    start = start.replace(",", ".").strip()
    parts = start.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def get_bvid(title: str) -> str:
    """根据纪录片名（解析后的 doc_name）返回 BVID。"""
    return _DOC_BVID_MAP.get(title, _PLACEHOLDER_BVID)


def build_bilibili_url(bvid: str, start_seconds: float) -> str:
    """构建 B站视频链接。"""
    t = int(start_seconds)
    return f"https://www.bilibili.com/video/{bvid}?t={t}"
