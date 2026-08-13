# 共通テスト

このディレクトリには、複数実験で共有するコードやリポジトリ全体の契約を検証するテストを置きます。

単一実験だけに属するテストは、対応する`experiments/<exp>/tests/`へ置きます。ルートで`task test`または`make test`を実行すると、Notebook依存を含む環境で両方の場所が収集・実行されます。

引数なしで`pytest`を直接実行した場合も、ルートの`conftest.py`が`project.yml.paths.experiments_dir`を読み、共通テストと設定後の実験ディレクトリを収集します。ファイルやディレクトリを明示した場合は、その指定範囲だけを収集します。

実験固有テストを収集せず、このディレクトリだけを確認する場合は`task test-common`または`make test-common`を使います。同梱skillの構造・任意のUI metadata・Pythonコードの静的検査は`task check-skills`または`make check-skills`で行います。
