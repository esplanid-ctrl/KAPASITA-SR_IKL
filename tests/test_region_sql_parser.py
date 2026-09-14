from app.data.region_sql_parser import parse_insert_rows


def test_single_row_per_statement():
    sql = "INSERT INTO `wilayah`(`kode`,`nama`) VALUES ('11','Aceh');\n" \
          "INSERT INTO `wilayah`(`kode`,`nama`) VALUES ('12','Sumatera Utara');\n"
    rows = parse_insert_rows(sql, expected_field_count=2)
    assert rows == [["11", "Aceh"], ["12", "Sumatera Utara"]]


def test_multi_row_values_list():
    sql = (
        "INSERT INTO `wilayah`(`kode`,`nama`) VALUES\n"
        "('11','Aceh'),\n"
        "('11.01','Kabupaten Aceh Selatan'),\n"
        "('11.02','Kabupaten Aceh Tenggara');\n"
    )
    rows = parse_insert_rows(sql, expected_field_count=2)
    assert rows == [
        ["11", "Aceh"],
        ["11.01", "Kabupaten Aceh Selatan"],
        ["11.02", "Kabupaten Aceh Tenggara"],
    ]


def test_names_with_commas_and_nested_brackets_do_not_break_field_split():
    sql = "INSERT INTO `t`(`kode`,`nama`,`path`) VALUES ('11','Test','[[1.0,2.0],[3.0,4.0]]');\n"
    rows = parse_insert_rows(sql, expected_field_count=3)
    assert rows[0][0] == "11"
    assert rows[0][2] == "[[1.0,2.0],[3.0,4.0]]"


def test_wrong_field_count_raises_instead_of_silently_truncating():
    sql = "INSERT INTO `t`(`a`,`b`) VALUES ('11','Aceh','extra');\n"
    try:
        parse_insert_rows(sql, expected_field_count=2)
        assert False, "should have raised"
    except ValueError:
        pass
