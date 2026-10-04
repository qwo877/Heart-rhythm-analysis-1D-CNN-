"""mitdb 的 AAMI / de Chazal inter-patient 切分。

DS1 訓練、DS2 測試，病人完全不重疊；val 從 DS1 再抽「整個病人」出來。
record 清單放 config，這裡負責把它切成 train / val / test 三組並做 sanity check。
"""

# 預設清單（也可由 config 覆寫）；起搏紀錄 102/104/107/217 本來就不在其中。
DS1_DEFAULT = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
               124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2_DEFAULT = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
               212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


def make_mitdb_split(cfg_data):
    """回傳 (train_records, val_records, test_records)，皆為 str list。"""
    ds1 = [str(r) for r in cfg_data.get("ds1", DS1_DEFAULT)]
    ds2 = [str(r) for r in cfg_data.get("ds2", DS2_DEFAULT)]
    val = [str(r) for r in cfg_data.get("val_records", [])]

    assert set(val).issubset(set(ds1)), "val_records 必須是 DS1 的子集（以病人為單位切 val）"
    train = [r for r in ds1 if r not in set(val)]
    return train, val, ds2
