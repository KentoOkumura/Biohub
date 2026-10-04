# 共通テスト

このディレクトリには、複数実験で共有するコードやリポジトリ全体の契約を検証するテストを置きます。

単一実験だけに属するテストは、対応する`experiments/<exp>/tests/`へ置きます。ルートで`task test`または`make test`を実行すると、Notebook・アプリ依存を含む環境で、共通テストと実験ごとのテストを別々のPythonプロセスで実行します。過去実験に同名のテスト・モジュールがあるため、全実験を単一のpytestプロセスへ集めません。失敗したグループがあっても残りを実行し、最後に失敗一覧を表示します。

引数なしの`pytest`は`pyproject.toml`に従い共通テストだけを収集します。全件検証の実験保存先は`scripts/run_tests.py`が`project.yml.paths.experiments_dir`から解決します。個別実験は`task test-exp EXP=<完全な実験名>`で実行します。

実験固有テストを収集せず、このディレクトリだけを確認する場合は`task test-common`または`make test-common`を使います。同梱skillの構造・任意のUI metadata・Pythonコードの静的検査は`task check-skills`または`make check-skills`で行います。
