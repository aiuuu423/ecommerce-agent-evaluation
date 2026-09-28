with bounds as (
  select
    max(date) as as_of_date,
    cast(max(date) - interval 29 day as date) as current_start,
    max(date) as current_end,
    cast(max(date) - interval 59 day as date) as previous_start,
    cast(max(date) - interval 30 day as date) as previous_end
  from traffic
),
metrics as (
  select
    coalesce(sum(revenue) filter (
      where order_date between current_start and current_end
    ), 0) as current_gmv,
    coalesce(count(distinct order_id) filter (
      where order_date between current_start and current_end
    ), 0) as current_orders,
    coalesce(sum(revenue) filter (
      where order_date between previous_start and previous_end
    ), 0) as previous_gmv,
    coalesce(count(distinct order_id) filter (
      where order_date between previous_start and previous_end
    ), 0) as previous_orders
  from bounds
  left join orders
    on order_date between previous_start and current_end
)
select
  as_of_date,
  current_start,
  current_end,
  previous_start,
  previous_end,
  current_gmv,
  previous_gmv,
  current_orders,
  previous_orders,
  current_gmv / nullif(current_orders, 0) as current_aov,
  previous_gmv / nullif(previous_orders, 0) as previous_aov,
  (current_gmv - previous_gmv) / nullif(previous_gmv, 0) as gmv_change_rate,
  (
    current_gmv / nullif(current_orders, 0)
    - previous_gmv / nullif(previous_orders, 0)
  ) / nullif(previous_gmv / nullif(previous_orders, 0), 0) as aov_change_rate
from metrics, bounds
order by as_of_date;
