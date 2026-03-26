import re

from src.utils.chromosome_vocab import canonicalize_chromosome_id


INVERSION_KARYOTYPE_BY_CHR = {
    "1": "inv(1)(p13q21)",
    "2": "inv(2)(p11.2q13)",
    "3": "inv(3)(q13.2q26.2)",
    "4": "inv(4)(q21.1q27)",
    "5": "inv(5)(q13.1q15)",
    "6": "inv(6)(p23q21)",
    "7": "inv(7)(p14q22)",
    "8": "inv(8)(p22p21.1)",
    "9": "inv(9)(q22.1q32)",
    "10": "inv(10)(p21q23)",
    "11": "inv(11)(q13.1q21)",
    "12": "inv(12)(p13.3q13.1)",
    "13": "inv(13)(q14q21)",
    "14": "inv(14)(p13q21)",
    "15": "inv(15)(q15q24)",
    "16": "inv(16)(p11.2q12.1)",
    "17": "inv(17)(p11.2q21.3)",
    "18": "inv(18)(p11.2q11.2)",
    "19": "inv(19)(p13.3p13.2)",
    "20": "inv(20)(p13q11.2)",
    "21": "inv(21)(q11.2q22.2)",
    "22": "inv(22)(q12.2q13.1)",
    "X": "inv(X)(p22.3p11.4)",
    "Y": "inv(Y)(p11.2q11.23)",
}


def _normalize_karyotype(text):
    return (
        str(text)
        .strip()
        .replace("（", "(")
        .replace("）", ")")
        .replace(" ", "")
    )


def _parse_breakpoint(text):
    match = re.fullmatch(r"([pq])(\d+)(?:\.(\d+))?", text)
    if not match:
        raise ValueError(f"Unsupported breakpoint format: {text}")

    arm = match.group(1)
    major = int(match.group(2))
    minor = match.group(3)
    major_token = f"{arm}{major}"

    return {
        "raw": text,
        "arm": arm,
        "arm_label": 0 if arm == "p" else 1,
        "major": major,
        "minor": int(minor) if minor is not None else -1,
        "major_token": major_token,
    }


def parse_inversion_karyotype(karyotype):
    normalized = _normalize_karyotype(karyotype)
    match = re.fullmatch(r"inv\(([^)]+)\)\(([pq]\d+(?:\.\d+)?)([pq]\d+(?:\.\d+)?)\)", normalized)
    if not match:
        raise ValueError(f"Unsupported inversion karyotype: {karyotype}")

    chr_id = canonicalize_chromosome_id(match.group(1))
    bp1 = _parse_breakpoint(match.group(2))
    bp2 = _parse_breakpoint(match.group(3))

    pericentric = int(bp1["arm"] != bp2["arm"])

    return {
        "chromosome_id": chr_id,
        "karyotype": normalized,
        "bp1": bp1["raw"],
        "bp2": bp2["raw"],
        "bp1_arm": bp1["arm"],
        "bp2_arm": bp2["arm"],
        "bp1_arm_label": bp1["arm_label"],
        "bp2_arm_label": bp2["arm_label"],
        "bp1_major_token": bp1["major_token"],
        "bp2_major_token": bp2["major_token"],
        "bp1_major": bp1["major"],
        "bp2_major": bp2["major"],
        "pericentric_label": pericentric,
        "paracentric_label": 1 - pericentric,
    }


_ALL_PARSED = [parse_inversion_karyotype(v) for v in INVERSION_KARYOTYPE_BY_CHR.values()]
BP_MAJOR_TOKENS = sorted(
    {row["bp1_major_token"] for row in _ALL_PARSED}.union({row["bp2_major_token"] for row in _ALL_PARSED})
)
BP_MAJOR_TOKEN_TO_IDX = {token: idx for idx, token in enumerate(BP_MAJOR_TOKENS)}
IDX_TO_BP_MAJOR_TOKEN = {idx: token for token, idx in BP_MAJOR_TOKEN_TO_IDX.items()}


def get_inversion_attributes_by_chromosome(chromosome_id):
    canon_chr = canonicalize_chromosome_id(chromosome_id)
    if canon_chr not in INVERSION_KARYOTYPE_BY_CHR:
        return None

    parsed = parse_inversion_karyotype(INVERSION_KARYOTYPE_BY_CHR[canon_chr]).copy()
    parsed["bp1_major_label"] = BP_MAJOR_TOKEN_TO_IDX[parsed["bp1_major_token"]]
    parsed["bp2_major_label"] = BP_MAJOR_TOKEN_TO_IDX[parsed["bp2_major_token"]]
    return parsed


def get_structure_label_dims():
    return {
        "pericentric": 2,
        "bp_arm": 2,
        "bp_major": len(BP_MAJOR_TOKENS),
    }
