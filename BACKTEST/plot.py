import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import webbrowser
import tempfile
import os
import numpy as np
import pandas as pd

class PlotlyChartFramework:
    def __init__(self):
        pass  # No need to store figures here since we generate full subplot

    def plot_performance(self, daily_reward_risk, cumulative_pnl, positions, drawdown, stats, title="Performance Chart", browser_name=None):
        x = cumulative_pnl.index
        y_cum_pnl = cumulative_pnl.iloc[:, 0]
        y_positions = positions.iloc[:, 0]
        y_drawdown = drawdown.iloc[:, 0]
        asset_name = drawdown.columns[0]
        # drawdown.index = drawdown.index.date
        daily_reward_risk.index = pd.to_datetime(daily_reward_risk.index)
        y_reward_risk = pd.concat([drawdown, daily_reward_risk], axis=1).iloc[:, 1]

        # --- Signals ---
        position_diff = positions.diff().iloc[:, 0]
        buy_signals = position_diff >= 1
        sell_signals = position_diff <= -1

        buy_times = x[buy_signals]
        sell_times = x[sell_signals]
        buy_prices = y_cum_pnl[buy_signals]
        sell_prices = y_cum_pnl[sell_signals]

        # --- Stats Table ---
        stat_names = list(stats.keys())
        stat_values = [stats[k][0] if isinstance(stats[k], list) else stats[k] for k in stat_names]
        stat_values = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in stat_values]

        # --- Make Subplots ---
        fig = make_subplots(
            rows=4, cols=1,
            row_heights=[0.5, 0.4, 0.4, 0.4],
            shared_xaxes=True,
            vertical_spacing=0.05,
            specs=[[{"type": "table"}], [{"secondary_y": True}], [{}], [{}]],
            subplot_titles=("Statistics Summary", "Cumulative PnL", "Daily Reward/Risk", "Drawdown")
        )

        # --- Table (Row 1) ---
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Statistic", "Value"],
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=12, color='black')
                ),
                cells=dict(
                    values=[stat_names, stat_values],
                    fill_color='lavender',
                    align='left',
                    font=dict(size=12)
                )
            ),
            row=1, col=1
        )

        # --- First subplot (Row 2) ---
        fig.add_trace(
            go.Scatter(x=x, y=y_cum_pnl, name='Cumulative PnL', line=dict(color='red'),
                connectgaps=True),
            row=2, col=1, secondary_y=False
        )
        # fig.add_trace(
        #     go.Bar(x=x, y=y_positions, name='Positions', marker=dict(color='blue'), opacity=0.5),
        #     row=2, col=1, secondary_y=True
        # )
        fig.add_trace(
            go.Scatter(x=buy_times, y=buy_prices, mode='markers', name='Buy',
                    marker=dict(symbol='triangle-up', color='green', size=10)),
            row=2, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=sell_times, y=sell_prices, mode='markers', name='Sell',
                    marker=dict(symbol='triangle-down', color='red', size=10)),
            row=2, col=1, secondary_y=False
        )
        
        # --- Second subplot (Row 3) ---
        # Keep only ONE point per day (place it at the LAST timestamp of each day)
        daily_rr = daily_reward_risk.squeeze().copy()
        # index by date (drop tz, strip time to date)
        daily_rr.index = pd.to_datetime(daily_rr.index).tz_localize(None).normalize()
        # if duplicate dates, keep the last (so the day's final value wins)
        daily_rr = daily_rr[~daily_rr.index.duplicated(keep='last')]

        x_naive = pd.to_datetime(x).tz_localize(None)
        x_dates = x_naive.normalize()

        # map each intraday timestamp's date -> daily value
        mapped = pd.Series(x_dates).map(daily_rr)

        # mark only the LAST timestamp of each day
        last_of_day = ~pd.Series(x_dates).duplicated(keep='last')

        # build y series: NaN everywhere except the last timestamp per day
        y_rr_series = pd.Series(index=pd.RangeIndex(len(x)), dtype="float64")
        y_rr_series[last_of_day.values] = mapped[last_of_day].values

        # running mean over the emitted daily points (align back to same index)
        y_mean_rr_0 = y_rr_series.expanding().mean()[y_rr_series.notna()]
        y_mean_rr = pd.Series(float('nan'), index=y_rr_series.index)
        y_mean_rr.loc[y_mean_rr_0.index] = y_mean_rr_0

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_rr_series.values,
                mode='lines+markers',   # one point (at day-end) per day
                name='Daily Reward/Risk',
                connectgaps=True
            ),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_mean_rr.values,
                mode='lines+markers',
                name='Mean Reward/Risk',
                connectgaps=True
            ),
            row=3, col=1
        )


        # --- Third subplot (Row 4) ---
        fig.add_trace(
            go.Scatter(x=x, y=y_drawdown, name='Drawdown', line=dict(color='red'),
                connectgaps=True),
            row=4, col=1
        )

        # --- Layout ---
        fig.update_layout(
            # height=900,
            title_text=f'{title} - Asset: {asset_name}',
            showlegend=True,
            margin=dict(t=60, b=40),
            autosize=True
        )

        fig.update_yaxes(title_text="Cumulative PnL", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Daily Reward/Risk", row=3, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Drawdown (in pct of max pnl)", row=4, col=1)
        fig.update_xaxes(title_text="Time", row=4, col=1)

        # --- Save HTML and Open in Browser (responsive height tied to width × rows) ---
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        pyo.plot(fig, filename=tmp_file.name, auto_open=False, config={'responsive': True})

        # Determine number of subplot rows from the figure
        n_rows = getattr(getattr(fig.layout, "grid", None), "rows", None)
        if n_rows is None:
            # fallback: count unique yaxis domains (works if using subplots)
            n_rows = sum(1 for k in fig.layout if str(k).startswith("yaxis") and getattr(fig.layout[k], "domain", None))

        # Tune this: height per row = ratioPerRow * window width
        ratio_per_row = 0.3  # e.g., 0.40 → each row is 0.4 × window width

        html = open(tmp_file.name, "r", encoding="utf-8").read()
        injection = f"""
        <style>
        html, body {{ margin:0; padding:0; height:100%; overflow:auto; }}
        </style>
        <script>
        (function() {{
        var rows = {int(n_rows) if n_rows else 1};
        var ratioPerRow = {ratio_per_row};

        function resizePlot() {{
            var w = window.innerWidth || document.documentElement.clientWidth;
            var targetH = Math.max(window.innerHeight, w * ratioPerRow * rows);
            var gd = document.querySelector('.plotly-graph-div');
            if (!gd) return;
            gd.style.width = '100%';
            gd.style.height = targetH + 'px';
            if (window.Plotly && Plotly.Plots && Plotly.Plots.resize) {{
            Plotly.Plots.resize(gd);
            }}
        }}

        window.addEventListener('load', resizePlot);
        window.addEventListener('resize', resizePlot);
        }})();
        </script>
        """
        html = html.replace("</head>", injection + "</head>")

        with open(tmp_file.name, "w", encoding="utf-8") as f:
            f.write(html)

        url = 'file://' + os.path.realpath(tmp_file.name)
        webbrowser.open(url)
    # self.plot_performance(cumulative_pnl, positions, drawdown, title=title, browser_name=browser_name)
