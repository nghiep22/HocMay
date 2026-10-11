"""Các bước offline: chuẩn bị, huấn luyện, đánh giá, hoặc toàn bộ."""
import sys as hethong
from chuanbidulieu import chuanbi

if hasattr(hethong.stdout, "reconfigure"):
    hethong.stdout.reconfigure(encoding="utf-8")


# Chọn và chạy một hoặc toàn bộ bước trong quy trình học máy.
def chayquytrinh():
    lenh = hethong.argv[1] if len(hethong.argv) > 1 else ""
    if lenh == "chuanbi":
        chuanbi()
    elif lenh == "huanluyen":
        from huanluyendanhgia import huanluyen
        huanluyen()
    elif lenh == "danhgia":
        from huanluyendanhgia import danhgia
        danhgia()
    elif lenh == "toanbo":
        from huanluyendanhgia import huanluyen, danhgia
        chuanbi()
        huanluyen()
        danhgia()
    else:
        raise SystemExit("Dùng: python chayquytrinh.py chuanbi|huanluyen|danhgia|toanbo")


if __name__ == "__main__":
    chayquytrinh()
