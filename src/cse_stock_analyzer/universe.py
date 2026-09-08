"""A diverse training universe of Colombo Stock Exchange securities."""

CSE_UNIVERSE = [
    ("COMB.N0000", "Commercial Bank of Ceylon"), ("HNB.N0000", "Hatton National Bank"),
    ("SAMP.N0000", "Sampath Bank"), ("NDB.N0000", "National Development Bank"),
    ("DFCC.N0000", "DFCC Bank"), ("SEYB.N0000", "Seylan Bank"),
    ("NTB.N0000", "Nations Trust Bank"), ("PABC.N0000", "Pan Asia Banking Corporation"),
    ("UBC.N0000", "Union Bank of Colombo"), ("LOLC.N0000", "LOLC Holdings"),
    ("LFIN.N0000", "LB Finance"), ("CFIN.N0000", "Central Finance"),
    ("PLC.N0000", "People's Leasing & Finance"), ("SFIN.N0000", "Senkadagala Finance"),
    ("AAIC.N0000", "Softlogic Life Insurance"), ("CINS.N0000", "Ceylinco Insurance"),
    ("JINS.N0000", "Janashakthi Insurance"), ("JKH.N0000", "John Keells Holdings"),
    ("HAYL.N0000", "Hayleys"), ("SPEN.N0000", "Aitken Spence"),
    ("RICH.N0000", "Richard Pieris"), ("VONE.N0000", "Vallibel One"),
    ("SOFT.N0000", "Softlogic Holdings"), ("EXPO.N0000", "Expolanka Holdings"),
    ("CARS.N0000", "Carson Cumberbatch"), ("BUKI.N0000", "Bukit Darah"),
    ("DIAL.N0000", "Dialog Axiata"), ("SLTL.N0000", "Sri Lanka Telecom"),
    ("TKYO.N0000", "Tokyo Cement"), ("RCL.N0000", "Royal Ceramics Lanka"),
    ("TILE.N0000", "Lanka Tiles"), ("ACL.N0000", "ACL Cables"),
    ("KZOO.N0000", "Kelani Cables"), ("DIPD.N0000", "Dipped Products"),
    ("CIC.N0000", "CIC Holdings"), ("GRAN.N0000", "Ceylon Grain Elevators"),
    ("LLUB.N0000", "Chevron Lubricants Lanka"), ("CTBL.N0000", "Ceylon Tobacco Company"),
    ("DIST.N0000", "Distilleries Company of Sri Lanka"), ("LION.N0000", "Lion Brewery Ceylon"),
    ("NEST.N0000", "Nestle Lanka"), ("CCS.N0000", "Ceylon Cold Stores"),
    ("MELS.N0000", "Melstacorp"), ("CARG.N0000", "Cargills Ceylon"),
    ("LIOC.N0000", "Lanka IOC"), ("LGL.N0000", "Laugfs Gas"),
    ("VPEL.N0000", "Vidullanka"), ("AHUN.N0000", "Aitken Spence Hotel Holdings"),
    ("KHL.N0000", "John Keells Hotels"), ("RHTL.N0000", "Renuka Hotels"),
    ("KVAL.N0000", "Kotagala Plantations"), ("WATA.N0000", "Watawala Plantations"),
    ("ASIR.N0000", "Asiri Hospital Holdings"), ("LHCL.N0000", "Lanka Hospitals"),
]


def normalize_symbol(symbol: str) -> str:
    """Normalize a CSE symbol to canonical form, e.g. ``JKH.N0000``."""
    value = symbol.strip().upper().removesuffix(".CM").replace("-", ".")
    if "." not in value:
        value += ".N0000"
    return value


def to_yahoo_symbol(symbol: str) -> str:
    """Convert a canonical CSE symbol to Yahoo Finance form."""
    return normalize_symbol(symbol).replace(".", "-") + ".CM"
