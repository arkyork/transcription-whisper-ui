
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from typing import Callable

class AudioProcess:
    # 音声→文字変換を行う
    # キューから音声データを取り出して文字起こしし、画面とファイルに出力
    @staticmethod
    def transcription_worker(
            q,
            model,
            SILENCE_THRESHOLD: int,
            update_ui: Callable[[str], None],
            transcript_file: Path, 
            stop_evt: threading.Event
        ):
        
        transcript_file.write_text("", encoding="utf-8")  # 出力ファイルを空にする

        while not stop_evt.is_set():
            try:
                audio = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if audio is None:
                break  # 終了信号

            if np.abs(audio).mean() < SILENCE_THRESHOLD:  # simple silence gate
                continue

            audio = audio.flatten().astype(np.float32)  # to 1-D float32 for model

            segments, _ = model.transcribe(audio, language="ja", beam_size=1)
            text = "".join(seg.text for seg in segments).strip()
            if not text:
                continue

            timestamp = datetime.now().strftime("%H:%M:%S")
            line = f"[{timestamp}] {text}"

            update_ui(line)  # 1) show on screen
            with transcript_file.open("a", encoding="utf-8") as f:  # 2) append file
                f.write(line + "\n")

    # 音声入力時に呼ばれる関数（非同期）

    @staticmethod
    def audio_callback(q, indata, frames, time_info, status):
        if status:
            print("⚠️ 音声入力エラー:", status)
        try:
            q.put_nowait(indata.copy())  # 音声をキューに追加
        except queue.Full:
            print("⚠️ キューが満杯のため、音声データを破棄しました")



