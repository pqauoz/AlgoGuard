import pytest

import train


def test_parser_accepts_the_documented_flags():
    args = train.build_parser().parse_args(["data.csv", "--deploy", "--admin", "analyst1"])
    assert args.csv_path == "data.csv"
    assert args.deploy is True
    assert args.admin == "analyst1"


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as exit_info:
        train.build_parser().parse_args(["--help"])
    assert exit_info.value.code == 0


def test_missing_csv_exits_non_zero(capsys):
    assert train.main(["definitely-not-a-real-dataset.csv"]) == 1
    assert "CSV file not found" in capsys.readouterr().err


def test_unknown_admin_exits_non_zero(tmp_path, binary_dataframe, capsys):
    csv_path = tmp_path / "traffic.csv"
    binary_dataframe.to_csv(csv_path, index=False)

    assert train.main([str(csv_path), "--admin", "nobody-here"]) == 1
    assert "was not found" in capsys.readouterr().err


def test_save_report_writes_the_run_csv(tmp_path):
    report_name = train.save_report(
        7,
        [
            {
                "rank": 1,
                "model_name": "Random Forest",
                "accuracy": 91.5,
                "normalized_metrics": {"accuracy": 0.915},
            }
        ],
        report_folder=str(tmp_path),
    )
    written = tmp_path / report_name
    assert report_name == "training_run_7_model_results.csv"
    assert written.exists()
    assert "Random Forest" in written.read_text()
