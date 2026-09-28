from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.data import generator
from app.data.config import load_data_config
from app.data.generator import (
    _anomaly_effects,
    _launch_date,
    _order_id,
    generate_customers,
    generate_dataset,
    generate_products,
)
from app.data.schemas import CustomerRow, MarketingRow, OrderRow, ProductRow, TrafficRow

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/data/synthetic_v1.yaml"


def test_dimension_generation_is_deterministic() -> None:
    config = load_data_config(CONFIG)

    assert_frame_equal(generate_products(config), generate_products(config))
    assert_frame_equal(generate_customers(config), generate_customers(config))


def test_different_seeds_change_random_dimension_fields() -> None:
    config = load_data_config(CONFIG)
    other_config = config.model_copy(update={"seed": config.seed + 1})

    products = generate_products(config)
    other_products = generate_products(other_config)
    customers = generate_customers(config)
    other_customers = generate_customers(other_config)

    for column in ("category", "price", "cost", "launch_date"):
        assert not products[column].equals(other_products[column])
    for column in ("is_new_customer", "region", "channel"):
        assert not customers[column].equals(other_customers[column])


def test_dimension_generation_is_independent_of_call_order() -> None:
    config = load_data_config(CONFIG)

    products_first = generate_products(config)
    customers_second = generate_customers(config)
    customers_first = generate_customers(config)
    products_second = generate_products(config)

    assert_frame_equal(products_first, products_second)
    assert_frame_equal(customers_first, customers_second)


@pytest.mark.parametrize(
    ("stream_name", "generated_table", "affected_columns"),
    [
        ("products.price", "products", {"price", "cost"}),
        ("products.margin", "products", {"cost"}),
        ("products.launch_date", "products", {"launch_date"}),
        ("products.category", "products", {"category"}),
        ("customers.is_new_customer", "customers", {"is_new_customer"}),
        ("customers.region", "customers", {"region"}),
        ("customers.channel", "customers", {"channel"}),
    ],
)
def test_random_fields_use_isolated_substreams(
    monkeypatch: pytest.MonkeyPatch,
    stream_name: str,
    generated_table: str,
    affected_columns: set[str],
) -> None:
    config = load_data_config(CONFIG)
    generate = generate_products if generated_table == "products" else generate_customers
    baseline = generate(config)
    original_rng = generator._rng

    def rng_with_one_changed_stream(current_config, current_stream_name):
        if current_stream_name == stream_name:
            changed_config = config.model_copy(update={"seed": config.seed + 1})
            return original_rng(changed_config, current_stream_name)
        return original_rng(current_config, current_stream_name)

    monkeypatch.setattr(generator, "_rng", rng_with_one_changed_stream)
    changed = generate(config)
    unaffected_columns = [
        column for column in baseline.columns if column not in affected_columns
    ]

    assert_frame_equal(baseline[unaffected_columns], changed[unaffected_columns])
    assert any(
        not baseline[column].equals(changed[column]) for column in affected_columns
    )


def test_dimension_ids_are_sequential_unique_and_counts_match() -> None:
    config = load_data_config(CONFIG)
    products = generate_products(config)
    customers = generate_customers(config)

    assert len(products) == config.product_count
    assert len(customers) == config.customer_count
    assert products["product_id"].tolist() == [
        f"P{number:03d}" for number in range(1, config.product_count + 1)
    ]
    assert customers["customer_id"].tolist() == [
        f"C{number:04d}" for number in range(1, config.customer_count + 1)
    ]
    assert products["product_id"].is_unique
    assert customers["customer_id"].is_unique


def test_dimensions_cover_configured_values_without_id_periodicity() -> None:
    config = load_data_config(CONFIG)
    products = generate_products(config)
    customers = generate_customers(config)

    assert set(products["category"]) == set(config.categories)
    assert set(customers["region"]) == set(config.regions)
    assert set(customers["channel"]) == set(config.channels)
    assert products["category"].tolist() != (
        config.categories * config.product_count
    )[: config.product_count]
    assert customers["region"].tolist() != (
        config.regions * config.customer_count
    )[: config.customer_count]
    assert customers["channel"].tolist() != (
        config.channels * config.customer_count
    )[: config.customer_count]


def test_v1_dimensions_are_balanced() -> None:
    config = load_data_config(CONFIG)
    category_counts = generate_products(config)["category"].value_counts()
    customers = generate_customers(config)
    region_counts = customers["region"].value_counts()
    channel_counts = customers["channel"].value_counts()

    assert category_counts.max() - category_counts.min() <= 1
    assert region_counts.max() - region_counts.min() <= 1
    assert channel_counts.max() - channel_counts.min() <= 1


def _has_fixed_period(values: list[str], period: int) -> bool:
    return all(value == values[index % period] for index, value in enumerate(values))


def test_balanced_dimensions_are_not_fixed_cycles_across_seeds() -> None:
    config = load_data_config(CONFIG)

    for seed in range(10):
        seeded_config = config.model_copy(update={"seed": seed})
        products = generate_products(seeded_config)
        customers = generate_customers(seeded_config)

        assert not _has_fixed_period(
            products["category"].tolist(), len(config.categories)
        )
        assert not _has_fixed_period(customers["region"].tolist(), len(config.regions))
        assert not _has_fixed_period(
            customers["channel"].tolist(), len(config.channels)
        )


def test_product_launch_dates_have_expected_range_and_diversity() -> None:
    config = load_data_config(CONFIG)
    launch_dates = generate_products(config)["launch_date"]

    assert launch_dates.min() >= config.start_date - timedelta(days=719)
    assert launch_dates.max() <= config.start_date - timedelta(days=30)
    assert launch_dates.nunique() > 1


@pytest.mark.parametrize(
    ("age_days", "expected"),
    [
        (30, date(2025, 12, 2)),
        (719, date(2024, 1, 13)),
    ],
)
def test_launch_date_pure_function_includes_age_boundaries(
    age_days: int, expected: date
) -> None:
    assert _launch_date(date(2026, 1, 1), age_days) == expected


def test_generated_dimensions_reuse_schema_and_config_contracts() -> None:
    config = load_data_config(CONFIG)
    products = generate_products(config)
    customers = generate_customers(config)

    validated_products = [
        ProductRow.model_validate(row) for row in products.to_dict(orient="records")
    ]
    for row in customers.to_dict(orient="records"):
        CustomerRow.model_validate(row)

    assert all(product.cost < product.price for product in validated_products)
    assert all(
        product.price.as_tuple().exponent >= -2
        and product.cost.as_tuple().exponent >= -2
        for product in validated_products
    )
    assert set(products["category"]) <= set(config.categories)
    assert set(customers["region"]) <= set(config.regions)
    assert set(customers["channel"]) <= set(config.channels)
    assert all(
        isinstance(value, Decimal) for value in products[["price", "cost"]].stack()
    )


def test_fact_tables_cover_the_complete_product_day_grid() -> None:
    config = load_data_config(CONFIG)
    tables = generate_dataset(config)
    expected_grid = pd.MultiIndex.from_product(
        [
            pd.date_range(config.start_date, periods=config.days).date,
            [f"P{number:03d}" for number in range(1, config.product_count + 1)],
        ],
        names=["date", "product_id"],
    )

    assert set(tables) == {"products", "customers", "traffic", "marketing", "orders"}
    for table_name in ("traffic", "marketing"):
        facts = tables[table_name]
        assert len(facts) == config.days * config.product_count
        assert not facts.duplicated(["date", "product_id"]).any()
        assert pd.MultiIndex.from_frame(facts[["date", "product_id"]]).equals(
            expected_grid
        )


def test_fact_generation_is_deterministic_with_unique_order_ids() -> None:
    config = load_data_config(CONFIG)
    first = generate_dataset(config)
    second = generate_dataset(config)

    for table_name in first:
        assert_frame_equal(first[table_name], second[table_name])
    assert first["orders"]["order_id"].is_unique
    assert first["orders"]["order_id"].str.fullmatch(
        r"O[0-9]{8}-P[0-9]{3}-[0-9]{6}"
    ).all()


def test_order_id_is_stable_and_supports_maximum_daily_sequence() -> None:
    assert _order_id(date(2026, 12, 31), "P999", 1) == "O20261231-P999-000001"
    assert _order_id(date(2026, 12, 31), "P999", 999_999) == (
        "O20261231-P999-999999"
    )


@pytest.mark.parametrize("sequence", [0, 1_000_000])
def test_order_id_rejects_sequence_outside_capacity(sequence: int) -> None:
    with pytest.raises(ValueError, match="order sequence"):
        _order_id(date(2026, 1, 1), "P001", sequence)


def test_fact_rows_reuse_declared_schemas() -> None:
    tables = generate_dataset(load_data_config(CONFIG))

    schema_by_table = {
        "traffic": TrafficRow,
        "marketing": MarketingRow,
        "orders": OrderRow,
    }
    for table_name, schema in schema_by_table.items():
        for row in tables[table_name].to_dict(orient="records"):
            schema.model_validate(row)


def test_fact_randomness_is_derived_from_field_date_product_and_order_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_data_config(CONFIG)
    stream_names: list[str] = []
    original_rng = generator._rng

    def recording_rng(current_config, stream_name):
        stream_names.append(stream_name)
        return original_rng(current_config, stream_name)

    monkeypatch.setattr(generator, "_rng", recording_rng)
    generate_dataset(config)

    fact_streams = [
        stream_name
        for stream_name in stream_names
        if stream_name.startswith(("traffic.", "orders."))
    ]
    assert any(
        stream_name.startswith("traffic.impressions|2026-01-01|P001")
        for stream_name in fact_streams
    )
    assert any(
        stream_name.startswith("orders.count|2026-01-01|P001")
        for stream_name in fact_streams
    )
    assert any(
        stream_name.startswith("orders.customer|2026-01-01|P001|")
        for stream_name in fact_streams
    )
    assert all(stream_name.count("|") >= 2 for stream_name in fact_streams)


def test_single_anomaly_counterfactual_is_isolated_with_the_same_seed() -> None:
    config = load_data_config(CONFIG)
    anomaly = next(item for item in config.anomalies if item.kind == "traffic_drop")
    isolated = config.model_copy(update={"anomalies": [anomaly]})
    counterfactual = config.model_copy(update={"anomalies": []})

    actual = generate_dataset(isolated)
    expected = generate_dataset(counterfactual)
    for table_name in ("products", "customers"):
        assert_frame_equal(actual[table_name], expected[table_name])

    start = config.start_date + timedelta(days=anomaly.start_day)
    end = config.start_date + timedelta(days=anomaly.end_day)
    for table_name, date_column in (
        ("traffic", "date"),
        ("marketing", "date"),
        ("orders", "order_date"),
    ):
        actual_rows = actual[table_name]
        expected_rows = expected[table_name]
        actual_unaffected = actual_rows[
            (actual_rows["product_id"] != anomaly.product_id)
            | ~actual_rows[date_column].between(start, end)
        ].reset_index(drop=True)
        expected_unaffected = expected_rows[
            (expected_rows["product_id"] != anomaly.product_id)
            | ~expected_rows[date_column].between(start, end)
        ].reset_index(drop=True)
        assert_frame_equal(actual_unaffected, expected_unaffected)

    common_orders = actual["orders"].merge(
        expected["orders"],
        on="order_id",
        suffixes=("_actual", "_expected"),
    )
    for field in ("customer_id", "quantity", "is_refund", "status"):
        assert (
            common_orders[f"{field}_actual"] == common_orders[f"{field}_expected"]
        ).all()


def test_missing_traffic_is_missing_observation_not_zero_business_activity() -> None:
    config = load_data_config(CONFIG)
    anomaly = next(item for item in config.anomalies if item.kind == "missing_traffic")
    isolated = config.model_copy(update={"anomalies": [anomaly]})
    counterfactual = config.model_copy(update={"anomalies": []})

    actual = generate_dataset(isolated)
    expected = generate_dataset(counterfactual)

    assert_frame_equal(actual["marketing"], expected["marketing"])
    assert_frame_equal(actual["orders"], expected["orders"])
    target = actual["traffic"][
        (actual["traffic"]["product_id"] == anomaly.product_id)
        & actual["traffic"]["date"].between(
            config.start_date + timedelta(days=anomaly.start_day),
            config.start_date + timedelta(days=anomaly.end_day),
        )
    ]
    assert target["is_missing"].all()
    assert target[["impressions", "clicks", "visits"]].isna().all().all()


def test_anomaly_window_uses_all_four_inclusive_boundaries() -> None:
    config = load_data_config(CONFIG)
    anomaly = config.anomalies[0]

    assert _anomaly_effects(config, anomaly.product_id, anomaly.start_day - 1)[
        "aov_multiplier"
    ] == pytest.approx(1.0)
    assert _anomaly_effects(config, anomaly.product_id, anomaly.start_day)[
        "aov_multiplier"
    ] == pytest.approx(anomaly.multiplier)
    assert _anomaly_effects(config, anomaly.product_id, anomaly.end_day)[
        "aov_multiplier"
    ] == pytest.approx(anomaly.multiplier)
    assert _anomaly_effects(config, anomaly.product_id, anomaly.end_day + 1)[
        "aov_multiplier"
    ] == pytest.approx(1.0)


def test_all_seven_configured_anomaly_kinds_map_to_expected_effects() -> None:
    config = load_data_config(CONFIG)
    expected_effects = {
        "sales_drop": {"aov_multiplier": 0.55},
        "traffic_drop": {"traffic_multiplier": 0.55},
        "conversion_drop": {"conversion_multiplier": 0.50},
        "high_refund": {"refund_multiplier": 4.00},
        "missing_traffic": {"missing_traffic": True},
        "extreme_traffic_spike": {"traffic_multiplier": 4.00},
        "multi_factor_drop": {
            "traffic_multiplier": 0.70,
            "conversion_multiplier": 0.70,
        },
    }

    assert {anomaly.kind for anomaly in config.anomalies} == set(expected_effects)
    for anomaly in config.anomalies:
        effects = _anomaly_effects(config, anomaly.product_id, anomaly.start_day)
        for effect, expected in expected_effects[anomaly.kind].items():
            assert effects[effect] == expected


def test_sales_drop_changes_only_gmv_and_aov_mechanism() -> None:
    config = load_data_config(CONFIG)
    anomaly = next(item for item in config.anomalies if item.kind == "sales_drop")
    actual = generate_dataset(config.model_copy(update={"anomalies": [anomaly]}))
    counterfactual = generate_dataset(config.model_copy(update={"anomalies": []}))
    start = config.start_date + timedelta(days=anomaly.start_day)
    end = config.start_date + timedelta(days=anomaly.end_day)

    actual_orders = actual["orders"][
        (actual["orders"]["product_id"] == anomaly.product_id)
        & actual["orders"]["order_date"].between(start, end)
    ]
    counterfactual_orders = counterfactual["orders"][
        (counterfactual["orders"]["product_id"] == anomaly.product_id)
        & counterfactual["orders"]["order_date"].between(start, end)
    ]

    assert_frame_equal(actual["traffic"], counterfactual["traffic"])
    assert actual_orders["order_id"].tolist() == counterfactual_orders["order_id"].tolist()
    assert actual_orders["quantity"].tolist() == counterfactual_orders["quantity"].tolist()
    assert actual_orders["revenue"].sum() < counterfactual_orders["revenue"].sum()
    assert actual_orders["revenue"].mean() < counterfactual_orders["revenue"].mean()


def test_configured_anomalies_are_observable_in_generated_facts() -> None:
    config = load_data_config(CONFIG)
    tables = generate_dataset(config)
    traffic = tables["traffic"]
    orders = tables["orders"]

    missing = traffic[(traffic["product_id"] == "P005") & traffic["is_missing"]]
    assert len(missing) == 6
    assert missing[["impressions", "clicks", "visits"]].isna().all().all()

    max_date = traffic["date"].max()
    current_start = max_date - timedelta(days=29)
    previous_start = max_date - timedelta(days=59)
    previous_end = max_date - timedelta(days=30)
    current_traffic = traffic[traffic["date"].between(current_start, max_date)]
    previous_traffic = traffic[traffic["date"].between(previous_start, previous_end)]
    current_orders = orders[orders["order_date"].between(current_start, max_date)]
    previous_orders = orders[orders["order_date"].between(previous_start, previous_end)]

    def visits(frame, product_id):
        return frame.loc[frame["product_id"] == product_id, "visits"].sum()

    def order_count(frame, product_id):
        return frame.loc[frame["product_id"] == product_id, "order_id"].nunique()

    def conversion(frame_orders, frame_traffic, product_id):
        return order_count(frame_orders, product_id) / visits(frame_traffic, product_id)

    current_p001 = current_orders[current_orders["product_id"] == "P001"]
    previous_p001 = previous_orders[previous_orders["product_id"] == "P001"]
    assert current_p001["revenue"].sum() < previous_p001["revenue"].sum()
    assert current_p001["revenue"].mean() < previous_p001["revenue"].mean()
    assert visits(current_traffic, "P002") < visits(previous_traffic, "P002")
    assert conversion(current_orders, current_traffic, "P003") < conversion(
        previous_orders, previous_traffic, "P003"
    )
    assert (
        current_orders.loc[current_orders["product_id"] == "P004", "is_refund"].mean()
        > previous_orders.loc[
            previous_orders["product_id"] == "P004", "is_refund"
        ].mean()
    )

    spike_date = config.start_date + timedelta(days=112)
    spike_impressions = traffic.loc[
        (traffic["product_id"] == "P006") & (traffic["date"] == spike_date),
        "impressions",
    ].item()
    normal_impressions = traffic.loc[
        (traffic["product_id"] == "P006")
        & traffic["date"].between(
            spike_date - timedelta(days=7), spike_date - timedelta(days=1)
        ),
        "impressions",
    ].median()
    assert spike_impressions > normal_impressions * 2

    assert visits(current_traffic, "P007") < visits(previous_traffic, "P007")
    assert order_count(current_orders, "P007") < order_count(previous_orders, "P007")
