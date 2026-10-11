"""Tiền xử lý cố định, không học quy tắc từ validation hoặc test."""
import re as bieuthuc

mauthu = bieuthuc.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", bieuthuc.I)
maulienket = bieuthuc.compile(r"https?://\S+", bieuthuc.I)
maudienthoai = bieuthuc.compile(r"\b(?:phone|tel|fax):\s*\+?[\d() ./-]{6,}\d\b", bieuthuc.I)


# Che thông tin liên hệ và chuẩn hóa khoảng trắng của văn bản.
def lamsach(noidung: str) -> str:
    """Loader sklearn đã bỏ headers/footers/quotes; tiếp tục che địa chỉ."""
    noidung = mauthu.sub("[thu da an]", noidung)
    noidung = maulienket.sub("[lien ket da an]", noidung)
    noidung = maudienthoai.sub("[dien thoai da an]", noidung)
    return bieuthuc.sub(r"\s+", " ", noidung).strip()


# Tách từ tiếng Anh theo cùng quy tắc dùng trong tìm kiếm.
def tachtu(noidung: str) -> list[str]:
    return bieuthuc.findall(r"[a-z0-9]+(?:'[a-z]+)?", noidung.lower())


# Lấy đoạn văn bản gần từ khóa để hiển thị trong kết quả.
def trichdoan(noidung: str, truyvan: str) -> str:
    noidung = lamsach(noidung)
    vitri = [noidung.lower().find(tu) for tu in set(tachtu(truyvan)) if len(tu) >= 3]
    vitri = [giatri for giatri in vitri if giatri >= 0]
    batdau = max(0, (min(vitri) if vitri else 0) - 70)
    ketthuc = min(len(noidung), batdau + 240)
    return ("…" if batdau else "") + noidung[batdau:ketthuc] + ("…" if ketthuc < len(noidung) else "")
