import gene_parser


def test_resolve_gene_value_options_preserves_dict():
    cfg = {"gene": "cl", "options": ["AND", "OR"]}
    assert gene_parser.resolve_gene_value(cfg) == "AND"
    assert cfg["options"] == ["AND", "OR"]


def test_resolve_gene_value_low_high():
    cfg = {"gene": "p", "low": 2, "high": 5}
    assert gene_parser.resolve_gene_value(cfg) == 2


def test_resolve_gene_value_high_only():
    cfg = {"gene": "p", "high": 5}
    assert gene_parser.resolve_gene_value(cfg) == 5


def test_resolve_gene_value_passthrough():
    assert gene_parser.resolve_gene_value(10) == 10
    assert gene_parser.resolve_gene_value("abc") == "abc"
