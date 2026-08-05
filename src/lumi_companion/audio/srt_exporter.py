import json
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import srt
from pydantic import BaseModel


class SubtitleSegment(BaseModel):
    """字幕セグメントデータモデル"""

    start: float  # 開始秒 (例: 1.5)
    end: float  # 終了秒 (例: 4.2)
    text: str  # 発言テキスト


class SRTExporter:
    """字幕 JSON および SRT ファイルの出力クラス"""

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """秒数を SRT 形式のタイムスタンプ文字列 (HH:MM:SS,mmm) に変換"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @classmethod
    def save_json(
        cls, segments: Sequence[SubtitleSegment], output_path: Path
    ) -> None:
        """字幕セグメントリストを JSON ファイルとして保存"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [segment.model_dump() for segment in segments]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def save_srt(
        cls, segments: Sequence[SubtitleSegment], output_path: Path
    ) -> None:
        """字幕セグメントリストを標準 .srt 字幕ファイルとして保存"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        srt_items: list[srt.Subtitle] = []

        for index, seg in enumerate(segments, start=1):
            srt_item = srt.Subtitle(
                index=index,
                start=timedelta(seconds=seg.start),
                end=timedelta(seconds=seg.end),
                content=seg.text,
            )
            srt_items.append(srt_item)

        formatted_srt = srt.compose(srt_items)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted_srt)
