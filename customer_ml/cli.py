"""Offline demo, training and inference command line interface."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from . import __version__
from .data import FILES, load_data, prepare_transactions, timestamp
from .demo import DEMO_CUTOFF, DEMO_DATA_THROUGH, make_demo
from .features import FEATURE_COLUMNS, TARGETS, build_dataset, create_features
from .model import feature_weights, train_and_evaluate
from .visualization import plot_feature_importance


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _source_arguments(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="使用明确标注的合成数据，无需联网")
    source.add_argument("--data-dir", type=Path, help="包含四张 Olist CSV 的目录")
    parser.add_argument("--cutoff", help="特征截止时间；不含此时刻，例如 2018-04-01")
    parser.add_argument("--seed", type=int, default=42)


def make_parser():
    parser = argparse.ArgumentParser(description="客户未来价值与复购预测（可复现学习项目）")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="构造时点特征，选模并评估")
    _source_arguments(train)
    train.add_argument("--data-through", help="声明完整订单覆盖的终点（不含）；真实数据必须指定")
    train.add_argument("--horizon-days", type=int, default=90)
    train.add_argument("--value-threshold", type=float, default=500.0)
    train.add_argument("--target", choices=TARGETS, default="high_value")
    train.add_argument("--output-dir", type=Path)
    predict = sub.add_parser("predict", help="用自己生成的可信模型对历史客户评分")
    _source_arguments(predict)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"))
    demo = sub.add_parser("demo-data", help="将合成数据与覆盖说明写入 CSV")
    demo.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    demo.add_argument("--customers", type=int, default=400)
    demo.add_argument("--seed", type=int, default=42)
    return parser


def _transactions(args):
    cutoff = args.cutoff or (DEMO_CUTOFF if args.demo else None)
    if cutoff is None:
        raise ValueError("真实数据必须显式指定 --cutoff。")
    timestamp(cutoff, "cutoff")
    tables = make_demo(seed=args.seed) if args.demo else load_data(args.data_dir)
    return prepare_transactions(tables), cutoff


def _train(args):
    transactions, cutoff = _transactions(args)
    data_through = args.data_through or (DEMO_DATA_THROUGH if args.demo else None)
    if data_through is None:
        raise ValueError("真实数据必须指定 --data-through；最大订单日期不能证明未来标签完整。")
    demo_csv = args.data_dir is not None and (args.data_dir / "DEMO_METADATA.json").is_file()
    if demo_csv:
        manifest = json.loads((args.data_dir / "DEMO_METADATA.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("data_source") != "synthetic_demo":
            raise ValueError("DEMO_METADATA.json 不是有效的本项目合成数据说明。")
        if manifest.get("data_through_exclusive") != DEMO_DATA_THROUGH:
            raise ValueError("DEMO_METADATA.json 的覆盖终点与合成器不符，请重新生成演示数据。")
    is_synthetic = args.demo or demo_csv
    if is_synthetic and timestamp(data_through) != timestamp(DEMO_DATA_THROUGH):
        raise ValueError(f"合成数据的真实覆盖终点固定为 {DEMO_DATA_THROUGH}，请勿修改覆盖声明。")
    dataset = build_dataset(transactions, cutoff, args.horizon_days, data_through, args.value_threshold)
    model, report, predictions, assignments = train_and_evaluate(dataset, args.target, args.seed)
    metadata = {
        "schema_version": 1, "project_version": __version__,
        "data_source": "synthetic_demo" if is_synthetic else "user_supplied_tables_unverified",
        "cutoff": str(timestamp(cutoff)), "label_end": str(timestamp(cutoff) + pd.Timedelta(days=args.horizon_days)),
        "data_through_exclusive": str(timestamp(data_through)),
        "horizon_days": args.horizon_days, "value_threshold": args.value_threshold,
        "amount_definition": "sum of item price; excludes freight, refunds and cancellations adjustment",
        "versions": report["versions"],
    }
    report.update(metadata)
    args.output_dir = args.output_dir or Path("artifacts") / args.target
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "metrics.json", report)
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    assignments.to_csv(args.output_dir / "customer_splits.csv", index=False)
    weights = feature_weights(model)
    weights.to_csv(args.output_dir / "feature_weights.csv", index=False)
    plot_feature_importance(weights, args.output_dir / "feature_weights.png")
    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "target": args.target,
                 "metadata": metadata}, args.output_dir / "model.joblib")
    print(f"数据来源：{report['data_source']}；客户 {len(dataset)}；目标 {args.target}")
    if is_synthetic:
        print("这是合成演示数据，指标只能验证流程，不能代表真实业务效果。")
    print(f"验证集选择：{report['selected_model']}；测试集 AP：{report['test']['average_precision']:.4f}")
    print(f"输出：{args.output_dir.resolve()}")


def _predict(args):
    transactions, cutoff = _transactions(args)
    features = create_features(transactions, cutoff)
    payload = joblib.load(args.model)
    if not isinstance(payload, dict) or payload.get("features") != FEATURE_COLUMNS or payload.get("metadata", {}).get("schema_version") != 1:
        raise ValueError("模型格式不匹配；请使用本项目 train 命令生成的模型。")
    result = features[["customer_unique_id"]].copy()
    result["probability"] = payload["model"].predict_proba(features[FEATURE_COLUMNS])[:, 1]
    result["prediction_at_0_5"] = (result["probability"] >= 0.5).astype(int)
    result["target"] = payload["target"]
    result["cutoff"] = str(timestamp(cutoff))
    result["horizon_days"] = payload["metadata"]["horizon_days"]
    result["training_data_source"] = payload["metadata"]["data_source"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"已为 {len(result)} 位客户评分：{args.output.resolve()}")


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            _train(args)
        elif args.command == "predict":
            _predict(args)
        else:
            tables = make_demo(args.customers, args.seed)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for name, table in tables.items():
                table.to_csv(args.output_dir / FILES[name], index=False)
            _write_json(args.output_dir / "DEMO_METADATA.json", {
                "data_source": "synthetic_demo", "customers": args.customers, "seed": args.seed,
                "cutoff": DEMO_CUTOFF, "data_through_exclusive": DEMO_DATA_THROUGH,
                "notice": "所有记录均由程序合成，并非 Olist 原始数据；不可作为真实业务效果证据。",
            })
            print(f"已生成明确标注的合成数据：{args.output_dir.resolve()}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"错误：{exc}\n")
    return 0
