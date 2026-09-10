import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QHBoxLayout, QFrame, QLabel)
from PyQt6.QtGui import QFont                    # ← 新增這行
from PyQt6.QtCore import Qt

class EpeeDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("擊劍多視圖")
        self.setGeometry(100, 100, 1980, 1080)

        background = QWidget()
        self.setCentralWidget(background)
        main_layout = QVBoxLayout(background)

        # 左(A)、中(B, D, C)、右(F)
        top_layout = QHBoxLayout()

        block_a = self.create_placeholder("Block A\n回合目錄\n", "#E8EAF6")
        block_a.setFixedWidth(250)
        top_layout.addWidget(block_a)

        # 中間直欄 (B, D, C)
        mid_layout = QVBoxLayout()
        
        block_b = self.create_placeholder("Block B\n戰術流向分析區", "#FFF9C4")
        #block_d = self.create_placeholder("Block D\n該回合影片播放區", "#212121", text_color="white")
        block_c = self.create_placeholder("Block C\n三軌並行視覺化列表", "#C8E6C9")

        mid_layout.addWidget(block_b, 5) 
        #mid_layout.addWidget(block_d, 5) 
        mid_layout.addWidget(block_c, 5) 
        
        top_layout.addLayout(mid_layout)

        # F: 右側控制面板
        block_f = self.create_placeholder("Block E \n 控制面板", "#FCE4EC")
        block_f.setFixedWidth(300)
        top_layout.addWidget(block_f)

        # 整個 A, B, C, D, F
        main_layout.addLayout(top_layout, 8) 

        # E: 累積得分階梯圖
        block_e = self.create_placeholder("Block D\n累積得分階梯圖", "#E0E0E0")
        main_layout.addWidget(block_e, 2) 

    def create_placeholder(self, text, bg_color, text_color="black"):
        """建立帶有背景顏色的佔位區塊（已套用論文正式字體）"""
        frame = QFrame()
        frame.setStyleSheet(f"background-color: {bg_color}; border: 1px solid #B0BEC5; border-radius: 5px;")
        layout = QVBoxLayout(frame)
        
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # === 正式論文字體設定 ===
        font = QFont()
        font.setFamilies(["標楷體", "Times New Roman", "serif"])  # 中文優先標楷體，英文優先 Times New Roman
        font.setPointSize(30)
        font.setBold(True)
        label.setFont(font)
        
        label.setStyleSheet(f"color: {text_color}; border: none;")
        layout.addWidget(label)
        return frame

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EpeeDashboard()
    window.show()
    sys.exit(app.exec())