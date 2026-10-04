"""afdb 的 record-level inter-patient 切分。

AF 是 subject-level 的病、整段標籤相同 → leakage 殺傷力比 beat 分類更大，
所以一定以「整筆 record（病人）」為單位切，不切 window。

afdb 沒有像 mitdb 那樣的官方 DS1/DS2；這裡先給一組固定、可重現的切分，
並在載入時印出每個 split 的 AF/non-AF window 數，若退化再調整。

註：00735 與 03665 兩筆在 afdb 沒有波形檔，多數文獻排除 → 預設不使用（23 筆）。
"""

AFDB_ALL = [
    "04015", "04043", "04048", "04126", "04746", "04908", "04936", "05091",
    "05121", "05261", "06426", "06453", "06995", "07162", "07859", "07879",
    "07910", "08215", "08219", "08378", "08405", "08434", "08455",
]
AFDB_EXCLUDE_DEFAULT = ["00735", "03665"]  # 無波形檔

# 預設 record-level 切分（病人不重疊）；可由 config 覆寫
TEST_DEFAULT = ["04048", "05121", "07162", "08405"]
VAL_DEFAULT = ["04126", "06453", "07879", "08434"]


def make_afdb_split(cfg_data):
    """回傳 (train_records, val_records, test_records)，皆為 str list、病人不重疊。"""
    all_records = [str(r) for r in cfg_data.get("records", AFDB_ALL)]
    exclude = {str(r) for r in cfg_data.get("exclude_records", AFDB_EXCLUDE_DEFAULT)}
    all_records = [r for r in all_records if r not in exclude]

    test = [str(r) for r in cfg_data.get("test_records", TEST_DEFAULT) if str(r) in all_records]
    val = [str(r) for r in cfg_data.get("val_records", VAL_DEFAULT) if str(r) in all_records]

    heldout = set(test) | set(val)
    assert not (set(test) & set(val)), "afdb val/test 病人重疊！"
    train = [r for r in all_records if r not in heldout]
    return train, val, test
