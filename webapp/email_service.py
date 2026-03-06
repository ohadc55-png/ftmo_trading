"""Weekly email report via Gmail SMTP."""

import logging
import os
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

from webapp.models import get_weekly_trades, _compute_stats

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def send_weekly_email():
    """Send weekly trade summary email. Called every Friday at 17:00 ET."""
    gmail_user = os.environ.get("GMAIL_USER")
    gmail_pass = os.environ.get("GMAIL_APP_PASSWORD")
    email_to = os.environ.get("EMAIL_TO")

    if not all([gmail_user, gmail_pass, email_to]):
        logger.warning("Email not configured (missing GMAIL_USER / GMAIL_APP_PASSWORD / EMAIL_TO)")
        return

    now = datetime.now(ET)
    week_end = now.strftime("%Y-%m-%d")
    week_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")

    trades = get_weekly_trades(week_start, week_end)
    stats = _compute_stats(trades)

    subject = f"NQ Paper Trading - Weekly Report ({week_start} to {week_end})"
    html_body = _build_email_html(stats, week_start, week_end)

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = gmail_user
        msg["To"] = email_to
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, email_to, msg.as_string())

        logger.info(f"Weekly email sent to {email_to}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}", exc_info=True)


def send_test_email():
    """Send a test email to verify configuration."""
    gmail_user = os.environ.get("GMAIL_USER")
    gmail_pass = os.environ.get("GMAIL_APP_PASSWORD")
    email_to = os.environ.get("EMAIL_TO")

    if not all([gmail_user, gmail_pass, email_to]):
        return False, "Missing env vars: GMAIL_USER, GMAIL_APP_PASSWORD, EMAIL_TO"

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "NQ Paper Trading Bot - Test Email"
        msg["From"] = gmail_user
        msg["To"] = email_to
        msg.attach(MIMEText(
            "<h2 style='color:#3fb950'>Email configuration works!</h2>"
            "<p>Your NQ Paper Trading Bot weekly reports will be sent to this address.</p>",
            "html"
        ))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, email_to, msg.as_string())

        return True, f"Test email sent to {email_to}"
    except Exception as e:
        return False, str(e)


def _build_email_html(stats: dict, week_start: str, week_end: str) -> str:
    """Build the weekly report email HTML with inline styles."""
    trades = stats.get("trades", [])
    n = stats["total_trades"]

    if n == 0:
        trades_summary = "<p style='color:#8b949e;'>No trades this week.</p>"
    else:
        trades_summary = f"""
        <p><strong>{n}</strong> trades |
        <span style="color:#3fb950">{stats['winners']}W</span> /
        <span style="color:#f85149">{stats['losers']}L</span> |
        Win Rate: <strong>{stats['win_rate']}%</strong></p>
        """

    pnl_color = "#3fb950" if stats["total_pnl"] >= 0 else "#f85149"

    # Build trade table rows
    trade_rows = ""
    for t in trades:
        dir_color = "#3fb950" if t["direction"] == "buy" else "#f85149"
        dir_label = "LONG" if t["direction"] == "buy" else "SHORT"
        pnl_c = "#3fb950" if t["pnl_dollars"] > 0 else "#f85149"
        outcome = t.get("outcome", "")

        trade_rows += f"""
        <tr style="border-bottom:1px solid #30363d;">
            <td style="padding:8px;color:#8b949e;font-size:12px;">{t['entry_time'][:16]}</td>
            <td style="padding:8px;">
                <span style="background:{'rgba(63,185,80,0.2)' if t['direction']=='buy' else 'rgba(248,81,73,0.2)'};
                    color:{dir_color};padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;">
                    {dir_label}
                </span>
            </td>
            <td style="padding:8px;font-family:monospace;font-size:12px;">{t['entry_price']:,.2f}</td>
            <td style="padding:8px;font-family:monospace;font-size:12px;">{(t.get('exit_price') or 0):,.2f}</td>
            <td style="padding:8px;font-size:12px;">{outcome}</td>
            <td style="padding:8px;color:{pnl_c};font-weight:600;font-family:monospace;">
                {'+'if t['pnl_dollars']>=0 else ''}${t['pnl_dollars']:,.0f}
            </td>
        </tr>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"></head>
    <body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
                 background:#0d1117;color:#e6edf3;padding:20px;margin:0;">
    <div style="max-width:700px;margin:0 auto;">

        <!-- Header -->
        <div style="text-align:center;padding:20px 0;border-bottom:1px solid #30363d;">
            <h1 style="color:#58a6ff;margin:0;font-size:24px;">NQ Paper Trading</h1>
            <p style="color:#8b949e;margin:8px 0 0;">Weekly Report: {week_start} to {week_end}</p>
            <p style="color:#8b949e;font-size:12px;margin:4px 0 0;">
                Strategy V3: BRK+MTF+VOL | RR 5.0 | 2c (1TP+1R) | SL&le;25 | SmartDL $750
            </p>
        </div>

        <!-- Summary -->
        <div style="padding:20px 0;">
            {trades_summary}
        </div>

        <!-- Key Stats Grid -->
        <div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:20px;">
            <div style="flex:1;min-width:120px;background:#161b22;border:1px solid #30363d;
                        border-radius:8px;padding:16px;">
                <div style="font-size:11px;color:#8b949e;text-transform:uppercase;">Total P&L</div>
                <div style="font-size:22px;font-weight:700;color:{pnl_color};">
                    {'+'if stats['total_pnl']>=0 else ''}${stats['total_pnl']:,.0f}
                </div>
            </div>
            <div style="flex:1;min-width:120px;background:#161b22;border:1px solid #30363d;
                        border-radius:8px;padding:16px;">
                <div style="font-size:11px;color:#8b949e;text-transform:uppercase;">Profit Factor</div>
                <div style="font-size:22px;font-weight:700;color:#58a6ff;">
                    {stats['profit_factor']:.2f}
                </div>
            </div>
            <div style="flex:1;min-width:120px;background:#161b22;border:1px solid #30363d;
                        border-radius:8px;padding:16px;">
                <div style="font-size:11px;color:#8b949e;text-transform:uppercase;">Win Rate</div>
                <div style="font-size:22px;font-weight:700;color:#58a6ff;">
                    {stats['win_rate']}%
                </div>
            </div>
            <div style="flex:1;min-width:120px;background:#161b22;border:1px solid #30363d;
                        border-radius:8px;padding:16px;">
                <div style="font-size:11px;color:#8b949e;text-transform:uppercase;">Max DD</div>
                <div style="font-size:22px;font-weight:700;color:#f85149;">
                    ${stats['max_drawdown']:,.0f}
                </div>
            </div>
        </div>

        <!-- Detailed Stats -->
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:20px;">
            <table style="width:100%;border-collapse:collapse;font-size:13px;">
                <tr>
                    <td style="padding:6px 0;color:#8b949e;">Avg Win</td>
                    <td style="padding:6px 0;color:#3fb950;text-align:right;font-weight:600;">
                        +${stats['avg_win']:,.0f}</td>
                </tr>
                <tr>
                    <td style="padding:6px 0;color:#8b949e;">Avg Loss</td>
                    <td style="padding:6px 0;color:#f85149;text-align:right;font-weight:600;">
                        ${stats['avg_loss']:,.0f}</td>
                </tr>
                <tr>
                    <td style="padding:6px 0;color:#8b949e;">Best Trade</td>
                    <td style="padding:6px 0;color:#3fb950;text-align:right;font-weight:600;">
                        +${stats['best_trade']:,.0f}</td>
                </tr>
                <tr>
                    <td style="padding:6px 0;color:#8b949e;">Worst Trade</td>
                    <td style="padding:6px 0;color:#f85149;text-align:right;font-weight:600;">
                        ${stats['worst_trade']:,.0f}</td>
                </tr>
                <tr>
                    <td style="padding:6px 0;color:#8b949e;">Expectancy</td>
                    <td style="padding:6px 0;text-align:right;font-weight:600;
                        color:{'#3fb950' if stats['expectancy']>=0 else '#f85149'};">
                        ${stats['expectancy']:,.0f}</td>
                </tr>
                <tr>
                    <td style="padding:6px 0;color:#8b949e;">Avg Bars Held</td>
                    <td style="padding:6px 0;text-align:right;color:#e6edf3;">
                        {stats['avg_bars_held']:.0f}</td>
                </tr>
                <tr>
                    <td style="padding:6px 0;color:#8b949e;">Max Consec Wins</td>
                    <td style="padding:6px 0;text-align:right;color:#3fb950;">{stats['max_consec_wins']}</td>
                </tr>
                <tr>
                    <td style="padding:6px 0;color:#8b949e;">Max Consec Losses</td>
                    <td style="padding:6px 0;text-align:right;color:#f85149;">{stats['max_consec_losses']}</td>
                </tr>
            </table>
        </div>

        <!-- Trade Log -->
        {'<h3 style="color:#e6edf3;margin-bottom:12px;">Trade Log</h3>' if trade_rows else ''}
        {'<table style="width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;border-radius:8px;font-size:13px;">' +
         '<thead><tr style="background:#1c2129;">' +
         '<th style="padding:10px 8px;text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;">Time</th>' +
         '<th style="padding:10px 8px;text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;">Dir</th>' +
         '<th style="padding:10px 8px;text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;">Entry</th>' +
         '<th style="padding:10px 8px;text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;">Exit</th>' +
         '<th style="padding:10px 8px;text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;">Outcome</th>' +
         '<th style="padding:10px 8px;text-align:left;color:#8b949e;font-size:11px;text-transform:uppercase;">P&L</th>' +
         '</tr></thead><tbody>' + trade_rows + '</tbody></table>' if trade_rows else ''}

        <!-- Footer -->
        <div style="margin-top:30px;padding:16px 0;border-top:1px solid #30363d;text-align:center;
                    color:#8b949e;font-size:11px;">
            NQ Futures Paper Trading Bot | Strategy V3 (RR 5.0)<br>
            Auto-generated weekly report
        </div>

    </div>
    </body>
    </html>
    """
