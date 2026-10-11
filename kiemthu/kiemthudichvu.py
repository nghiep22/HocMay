import pytest as kiemthu
from pathlib import Path as duongdan
from tempfile import TemporaryDirectory as thumuctam
from fastapi.testclient import TestClient as khachkiemthu

from xephangtimkiem import taomohinh, botimkiem
from dichvuapi import taodichvuapi
import dichvuapi as moduldichvu


baiviet = [
    {"madinhdanh": "train/sci.space/a", "chude": "sci.space",
     "noidung": "rocket launch satellite mission", "tap": "huanluyen"},
    {"madinhdanh": "train/rec.autos/b", "chude": "rec.autos",
     "noidung": "car engine road repair", "tap": "huanluyen"},
]
cauhinh = {"minDf": 1, "maxDf": 1.0, "ngramMax": 1}


# Kiểm tra chỉ học từ train và xếp hạng bài có từ khóa phù hợp.
def test_mohinh_chi_fit_train_va_xep_hang():
    mohinh = taomohinh(baiviet, cauhinh)
    timkiem = botimkiem(mohinh)
    ketqua = timkiem.timbaiviet("rocket launch", 2)
    assert ketqua[0]["id"] == baiviet[0]["madinhdanh"]
    assert ketqua[0]["contributingTerms"]
    assert not timkiem.timbaiviet("unseenword", 2)
    with kiemthu.raises(ValueError, match="huấn luyện"):
        taomohinh(baiviet + [{**baiviet[0], "tap": "xacthuc"}], cauhinh)


# Kiểm tra mã lỗi và cấu trúc phản hồi của các API chính.
def test_api_kiemtra_dauvao_va_phanhoi():
    khach = khachkiemthu(taodichvuapi(taomohinh(baiviet, cauhinh)))
    assert khach.get("/api/search", params={"q": "x", "k": 100}).status_code == 400
    assert khach.get("/api/search", params={"q": "rocket", "k": "9" * 5000}).status_code == 400
    assert khach.get("/api/search", params={"q": "rocket", "category": "unknown"}).status_code == 400
    phanhoi = khach.get("/api/search", params={"q": "rocket launch", "k": 2})
    assert phanhoi.status_code == 200
    assert phanhoi.json()["data"][0]["category"] == "sci.space"
    assert phanhoi.json()["meta"]["responseMs"] >= 0
    assert khach.get("/api/meta").json()["data"]["documents"] == 2
    assert khach.get("/api/does-not-exist").status_code == 404


# Kiểm tra API báo thiếu mô hình khi tệp chưa được tạo.
def test_api_bao_thieu_mohinh():
    with thumuctam() as tenthumuc, kiemthu.MonkeyPatch.context() as suadoi:
        suadoi.setattr(moduldichvu, "tepmohinh", duongdan(tenthumuc) / "khongco.joblib")
        khach = khachkiemthu(taodichvuapi())
        assert khach.get("/api/health").status_code == 503
        assert khach.get("/api/search", params={"q": "rocket"}).status_code == 503
