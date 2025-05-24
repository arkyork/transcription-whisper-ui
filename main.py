

import flet as ft
from scripts.ui import WhisperUI

app = WhisperUI()

if __name__ == "__main__":
    ft.app(target=app.main)
