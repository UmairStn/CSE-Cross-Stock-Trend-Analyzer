from cse_stock_analyzer.universe import normalize_symbol, to_yahoo_symbol


def test_symbol_normalization():
    assert normalize_symbol("jkh") == "JKH.N0000"
    assert normalize_symbol("jkh-n0000.cm") == "JKH.N0000"
    assert to_yahoo_symbol("JKH.N0000") == "JKH-N0000.CM"
