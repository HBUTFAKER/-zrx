"""探测：退市股(zs.300089 文化退)在新浪/其他源是否还有财报数据。"""
import warnings

warnings.filterwarnings("ignore")

CODES = ["300089", "300356", "002680"]  # 全是已退市


def show(name, fn):
    print(f"\n===== {name} =====", flush=True)
    for c in CODES:
        try:
            df = fn(c)
            if df is None or len(df) == 0:
                print(f"  {c}: EMPTY")
                continue
            print(f"  {c}: shape={df.shape}")
            print(f"    cols={list(df.columns)[:14]}")
            print(f"    head={df.head(1).to_dict('records')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {c}: EXC {type(e).__name__}: {str(e)[:90]}")


import akshare as ak  # noqa: E402

show("stock_financial_abstract（新浪关键指标）", lambda c: ak.stock_financial_abstract(symbol=c))
show("stock_financial_analysis_indicator（新浪财务指标）",
     lambda c: ak.stock_financial_analysis_indicator(symbol=c, start_year="2018"))
show("stock_financial_report_sina 利润表",
     lambda c: ak.stock_financial_report_sina(stock=f"sz{c}", symbol="利润表"))
