"""mitdb beat annotation → AAMI 五分類（Phase 0 沿用）。"""

# AAMI 五分類：0=N, 1=S(SVEB), 2=V(VEB), 3=F(Fusion), 4=Q(Unknown/Paced)
CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def build_symbol_map():
    """只保留真正的 beat annotation；非心跳標記查不到就由呼叫端 skip。"""
    return {
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
        "A": 1, "a": 1, "J": 1, "S": 1,
        "V": 2, "E": 2,
        "F": 3,
        "/": 4, "f": 4, "Q": 4,
    }


def num_classes_from_map(sym_map=None):
    """類別數由 symbol map 推導，不寫死。"""
    sym_map = sym_map or build_symbol_map()
    return len(set(sym_map.values()))
