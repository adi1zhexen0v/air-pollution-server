import nodemailer from "nodemailer";
import "dotenv/config";

const FONT_FAMILY =
  "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
const DASHBOARD_URL = "https://thesis.portfolio-adilzhexenov.kz";

const transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: process.env.NODEMAILER_EMAIL,
    pass: process.env.NODEMAILER_PASSWORD,
  },
});

function formatGMT5(date) {
  const d = new Date(date);
  return d.toLocaleString("en-US", {
    timeZone: "Asia/Almaty",
    weekday: "short",
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
}

function buildAlertCard(alert) {
  const isCritical = alert.severity === "critical";
  const cardBg = isCritical ? "#fcebeb" : "#faeeda";
  const cardBorder = isCritical ? "#f09595" : "#fac775";
  const dotColor = isCritical ? "#e24b4a" : "#854f0b";

  let detailRows = "";
  if (alert.details && alert.details.length > 0) {
    const rows = alert.details
      .map((d) => {
        const valueStyle = [
          `color: ${d.color || "#1a1a2e"}`,
          `font-size: 14px`,
          d.bold ? "font-weight: 700" : "font-weight: 400",
        ].join("; ");

        return `<tr>
          <td style="padding: 4px 12px 4px 0; color: #888888; font-size: 13px; font-family: ${FONT_FAMILY}; white-space: nowrap; vertical-align: top;">${d.label}</td>
          <td style="${valueStyle}; font-family: ${FONT_FAMILY}; vertical-align: top;">${d.value}</td>
        </tr>`;
      })
      .join("");

    detailRows = `
      <tr><td colspan="2" style="padding: 10px 0 6px 0;">
        <div style="border-top: 1px solid #e8e8e8;"></div>
      </td></tr>
      ${rows}`;
  }

  return `
    <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom: 12px;">
      <tr><td style="padding: 16px; background-color: ${cardBg}; border: 1px solid ${cardBorder}; border-radius: 8px;">
        <table width="100%" cellpadding="0" cellspacing="0" border="0">
          <tr><td style="padding-bottom: 6px;">
            <span style="color: ${dotColor}; font-size: 18px; line-height: 1; vertical-align: middle;">&#9679;</span>
            <span style="font-size: 15px; font-weight: 700; color: #1a1a2e; font-family: ${FONT_FAMILY}; vertical-align: middle; padding-left: 6px;">${alert.title}</span>
          </td></tr>
          <tr><td style="padding-bottom: 4px; font-size: 14px; color: #555555; font-family: ${FONT_FAMILY}; line-height: 1.5;">
            ${alert.description}
          </td></tr>
          ${detailRows ? `<tr><td><table width="100%" cellpadding="0" cellspacing="0" border="0">${detailRows}</table></td></tr>` : ""}
        </table>
      </td></tr>
    </table>`;
}

function buildSeveritySection(label, alerts, labelColor) {
  if (alerts.length === 0) return "";

  const cards = alerts.map((a) => buildAlertCard(a)).join("");

  return `
    <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom: 8px;">
      <tr><td style="padding: 16px 0 8px 0;">
        <span style="display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: 1px; color: ${labelColor}; font-family: ${FONT_FAMILY}; text-transform: uppercase;">${label}</span>
      </td></tr>
    </table>
    ${cards}`;
}

function buildActionCard(actionText) {
  if (!actionText) return "";

  return `
    <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom: 8px;">
      <tr><td style="padding: 16px 0 8px 0;">
        <span style="display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: 1px; color: #854f0b; font-family: ${FONT_FAMILY}; text-transform: uppercase;">RECOMMENDED ACTION</span>
      </td></tr>
    </table>
    <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom: 16px;">
      <tr><td style="padding: 14px 16px; background-color: #faeeda; border: 1px solid #fac775; border-radius: 8px; font-size: 14px; color: #555555; font-family: ${FONT_FAMILY}; line-height: 1.5;">
        ${actionText}
      </td></tr>
    </table>`;
}

export function buildAlertEmail(alerts, timestamp) {
  const ts = new Date(timestamp);
  const timeStr = formatGMT5(ts);

  const critical = alerts.filter((a) => a.severity === "critical");
  const warning = alerts.filter((a) => a.severity === "warning");

  const firstAction = alerts.find((a) => a.action)?.action || null;

  const criticalSection = buildSeveritySection("CRITICAL", critical, "#a32d2d");
  const warningSection = buildSeveritySection("WARNINGS", warning, "#854f0b");

  return `<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin: 0; padding: 0; background-color: #f0f0ea; font-family: ${FONT_FAMILY};">
  <table width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color: #f0f0ea; padding: 24px 0;">
    <tr><td align="center">
      <table width="560" cellpadding="0" cellspacing="0" border="0" style="max-width: 560px; width: 100%; border: 1px solid #e0e0da; border-radius: 12px; overflow: hidden;">

        <!-- HEADER -->
        <tr><td style="background-color: #1a1a2e; padding: 24px 28px;">
          <table width="100%" cellpadding="0" cellspacing="0" border="0">
            <tr>
              <td style="vertical-align: top;">
                <div style="font-size: 20px; font-weight: 700; color: #ffffff; font-family: ${FONT_FAMILY}; padding-bottom: 4px;">&#9888;&#65039; Sensor alert</div>
                <div style="font-size: 13px; color: #9a9ab0; font-family: ${FONT_FAMILY};">Astana air quality monitoring system</div>
              </td>
              <td style="vertical-align: top; text-align: right;">
                <div style="font-size: 13px; color: #9a9ab0; font-family: ${FONT_FAMILY}; white-space: nowrap;">${timeStr}</div>
                <div style="font-size: 11px; color: #9a9ab0; font-family: ${FONT_FAMILY};">(GMT+5)</div>
              </td>
            </tr>
          </table>
        </td></tr>

        <!-- BODY -->
        <tr><td style="background-color: #ffffff; padding: 8px 28px 24px 28px;">
          ${criticalSection}
          ${warningSection}
          ${buildActionCard(firstAction)}

          <!-- DASHBOARD BUTTON -->
          <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-top: 8px;">
            <tr><td align="center" style="padding: 12px 0 4px 0;">
              <a href="${DASHBOARD_URL}" target="_blank" style="display: inline-block; padding: 12px 32px; background-color: #1a1a2e; color: #ffffff; font-size: 14px; font-weight: 600; font-family: ${FONT_FAMILY}; text-decoration: none; border-radius: 8px;">Open dashboard</a>
            </td></tr>
          </table>
        </td></tr>

        <!-- FOOTER -->
        <tr><td style="background-color: #f5f5f0; padding: 18px 28px; text-align: center;">
          <div style="font-size: 12px; color: #999999; font-family: ${FONT_FAMILY}; line-height: 1.5;">Automated alert from Astana Air Quality Monitoring System</div>
        </td></tr>

      </table>
    </td></tr>
  </table>
</body>
</html>`;
}

function buildSubject(alerts) {
  const critical = alerts.filter((a) => a.severity === "critical");
  const warning = alerts.filter((a) => a.severity === "warning");

  if (critical.length > 0) {
    return `🔴 CRITICAL: ${critical[0].title} | Astana AQ Monitor`;
  }
  return `⚠️ WARNING: ${warning[0].title} | Astana AQ Monitor`;
}

export async function sendAlertEmail(alerts, timestamp) {
  try {
    const html = buildAlertEmail(alerts, timestamp);
    const subject = buildSubject(alerts);

    const info = await transporter.sendMail({
      from: `"Astana AQ Monitor" <${process.env.NODEMAILER_EMAIL}>`,
      to: process.env.ALERT_EMAIL_TO,
      subject,
      html,
    });

    console.log(`[INFO] Alert email sent to ${process.env.ALERT_EMAIL_TO} — MessageID: ${info.messageId}`);
    return { success: true, messageId: info.messageId };
  } catch (error) {
    console.error("[ERR] Failed to send alert email:", error.message);
    return { success: false, error: error.message };
  }
}
