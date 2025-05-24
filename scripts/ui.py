import json
import queue
import threading
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import flet as ft
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel

from .audio_process import AudioProcess



class WhisperUI:
    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        self.q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=50)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        print("使用デバイス:", self.device, "→ 計算タイプ:", self.compute_type)

        print("Whisperモデルを読み込み中...")
        self.model = WhisperModel(
            self.config["model_name"],
            device=self.device,
            compute_type=self.compute_type,
        )
        print("モデル準備完了 →", self.config["model_name"])

        # 状態保持
        self.stop_evt = threading.Event()
        self.stream_thread_ref: list[threading.Thread | None] = [None]

        self.devices = []
        self.input_devices = []
        
        all_devices = sd.query_devices()
        seen = set()
        self.input_devices = []
        for i, d in enumerate(all_devices):
            name = d["name"]
            if d["max_input_channels"] > 0 and name not in seen:
                self.input_devices.append((i, name))
                seen.add(name)
        self.devices = all_devices

    def load_config(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def main(self, page: ft.Page):
        page.title = "リアルタイム議事録生成"
        page.scroll = "AUTO"

        self.subtitle = ft.Text("＜字幕をここに表示します＞", selectable=True, size=20)
        page.add(self.subtitle)

        def update_ui(new_line: str):
            self.subtitle.value = new_line
            page.update()

        self.update_ui = update_ui

        self.filename_field = ft.TextField(label="出力ファイル名", value=self.config["output_file"], width=400)

        self.device_dropdown = ft.Dropdown(
            label="音声入力デバイスを選択",
            options=[ft.dropdown.Option(str(i), text=name) for i, name in self.input_devices],
            width=500,
        )

        self.start_button = ft.ElevatedButton(text="文字起こし開始", disabled=True)
        self.stop_button = ft.ElevatedButton(text="停止", disabled=True, bgcolor="#f87171")
        self.status = ft.Text("デバイスを選んでください", color="gray")

        # イベント登録
        self.device_dropdown.on_change = self.on_device_change
        self.start_button.on_click = self.on_start_click
        self.stop_button.on_click = self.on_stop_click

        # UIレイアウト追加
        page.add(
            self.filename_field,
            self.device_dropdown,
            ft.Row([self.start_button, self.stop_button]),
            self.status
        )

    def on_device_change(self, e):
        self.start_button.disabled = self.device_dropdown.value is None
        if self.device_dropdown.value is not None:
            text = self.devices[int(self.device_dropdown.value)]["name"]
            self.status.value = f"選択中: {text}"
        self.start_button.page.update()

    def on_start_click(self, e):
        

        if self.device_dropdown.value is None:
            return

        self.stop_evt.clear()
        device_id = int(self.device_dropdown.value)

        file_name = self.filename_field.value.strip() or "transcription.txt"
        transcript_path = Path("outputs") / file_name
        transcript_path.parent.mkdir(parents=True, exist_ok=True)  # フォルダがなければ作る        
        self.q = queue.Queue(maxsize=50)

        print("🎤 録音開始:", self.devices[device_id]["name"], "→ 出力:", transcript_path)

        def stream_thread():
            worker = threading.Thread(
                target=AudioProcess.transcription_worker,
                args=(self.q, self.model , self.config["silence_threshold"] , self.update_ui, transcript_path, self.stop_evt),
                daemon=True,
            )
            worker.start()

            with sd.InputStream(
                channels=1,
                callback=partial(AudioProcess.audio_callback, self.q), 
                samplerate=self.config["sample_rate"],
                blocksize=int(self.config["sample_rate"] * self.config["block_sec"]),
                device=device_id,
            ):
                self.status.value = (
                    f"🎤 入力デバイス: {self.devices[device_id]['name']}（録音中）…停止ボタンで終了"
                )
                self.start_button.page.update()
                while not self.stop_evt.is_set():
                    sd.sleep(200)

            self.q.put(None)
            worker.join()
            self.status.value = "⏸️ 録音停止。再開できます。"
            self.start_button.page.update()

        t = threading.Thread(target=stream_thread, daemon=True)
        t.start()
        self.stream_thread_ref[0] = t

        self.start_button.disabled = True
        self.stop_button.disabled = False
        self.filename_field.disabled = True
        self.device_dropdown.disabled = True
        self.start_button.page.update()

    def on_stop_click(self, e):
        self.stop_button.disabled = True
        self.stop_evt.set()
        t = self.stream_thread_ref[0]
        if t is not None:
            t.join(timeout=1.0)
        self.start_button.disabled = False
        self.filename_field.disabled = False
        self.device_dropdown.disabled = False
        self.start_button.page.update()


