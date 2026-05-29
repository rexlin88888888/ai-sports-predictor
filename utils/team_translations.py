from __future__ import annotations

from functools import lru_cache
import re


FOOTBALL_TEAM_TRANSLATIONS_ZH = {
    "Algeria": "阿尔及利亚",
    "Argentina": "阿根廷",
    "Australia": "澳大利亚",
    "Austria": "奥地利",
    "Belgium": "比利时",
    "Brazil": "巴西",
    "Cameroon": "喀麦隆",
    "Canada": "加拿大",
    "Chile": "智利",
    "China": "中国",
    "Colombia": "哥伦比亚",
    "Costa Rica": "哥斯达黎加",
    "Croatia": "克罗地亚",
    "Czech Republic": "捷克",
    "Czechia": "捷克",
    "Denmark": "丹麦",
    "Ecuador": "厄瓜多尔",
    "Egypt": "埃及",
    "England": "英格兰",
    "France": "法国",
    "Germany": "德国",
    "Ghana": "加纳",
    "Greece": "希腊",
    "Iran": "伊朗",
    "Iraq": "伊拉克",
    "Italy": "意大利",
    "Ivory Coast": "科特迪瓦",
    "Japan": "日本",
    "Korea Republic": "韩国",
    "Mexico": "墨西哥",
    "Morocco": "摩洛哥",
    "Netherlands": "荷兰",
    "New Zealand": "新西兰",
    "Nigeria": "尼日利亚",
    "Norway": "挪威",
    "Paraguay": "巴拉圭",
    "Poland": "波兰",
    "Portugal": "葡萄牙",
    "Qatar": "卡塔尔",
    "Saudi Arabia": "沙特阿拉伯",
    "Senegal": "塞内加尔",
    "Serbia": "塞尔维亚",
    "South Africa": "南非",
    "Spain": "西班牙",
    "Sweden": "瑞典",
    "Switzerland": "瑞士",
    "Tunisia": "突尼斯",
    "Turkey": "土耳其",
    "Turkiye": "土耳其",
    "United States": "美国",
    "Uruguay": "乌拉圭",
    "Uzbekistan": "乌兹别克斯坦",
    "Venezuela": "委内瑞拉",
    "Wales": "威尔士",
    "Cape Verde": "佛得角",
    "Congo DR": "刚果民主共和国",
    "DR Congo": "刚果民主共和国",
    "Democratic Republic of the Congo": "刚果民主共和国",
}


NBA_TEAM_TRANSLATIONS_ZH = {
    "Atlanta Hawks": "亚特兰大老鹰",
    "Boston Celtics": "波士顿凯尔特人",
    "Brooklyn Nets": "布鲁克林篮网",
    "Charlotte Hornets": "夏洛特黄蜂",
    "Chicago Bulls": "芝加哥公牛",
    "Cleveland Cavaliers": "克利夫兰骑士",
    "Dallas Mavericks": "达拉斯独行侠",
    "Denver Nuggets": "丹佛掘金",
    "Detroit Pistons": "底特律活塞",
    "Golden State Warriors": "金州勇士",
    "Houston Rockets": "休斯顿火箭",
    "Indiana Pacers": "印第安纳步行者",
    "LA Clippers": "洛杉矶快船",
    "Los Angeles Clippers": "洛杉矶快船",
    "Los Angeles Lakers": "洛杉矶湖人",
    "Memphis Grizzlies": "孟菲斯灰熊",
    "Miami Heat": "迈阿密热火",
    "Milwaukee Bucks": "密尔沃基雄鹿",
    "Minnesota Timberwolves": "明尼苏达森林狼",
    "New Orleans Pelicans": "新奥尔良鹈鹕",
    "New York Knicks": "纽约尼克斯",
    "Oklahoma City Thunder": "俄克拉荷马城雷霆",
    "Orlando Magic": "奥兰多魔术",
    "Philadelphia 76ers": "费城76人",
    "Phoenix Suns": "菲尼克斯太阳",
    "Portland Trail Blazers": "波特兰开拓者",
    "Sacramento Kings": "萨克拉门托国王",
    "San Antonio Spurs": "圣安东尼奥马刺",
    "Toronto Raptors": "多伦多猛龙",
    "Utah Jazz": "犹他爵士",
    "Washington Wizards": "华盛顿奇才",
}


SPECIAL_TRANSLATIONS_ZH = {
    "Draw": "平局",
    "DRAW": "平局",
    "Home": "主队",
    "Away": "客队",
}


TEAM_TRANSLATIONS_ZH = {
    **FOOTBALL_TEAM_TRANSLATIONS_ZH,
    **NBA_TEAM_TRANSLATIONS_ZH,
    **SPECIAL_TRANSLATIONS_ZH,
}

TEAM_TRANSLATIONS_EN = {value: key for key, value in TEAM_TRANSLATIONS_ZH.items()}

UI_TEXT_ZH = {
    "FOOTBALL": "足球",
    "Football": "足球",
    "football": "足球",
    "NBA": "NBA",
    "Home": "主队",
    "Away": "客队",
    "Confidence": "信心指数",
    "LOW": "低",
    "Low": "低",
    "low": "低",
    "MEDIUM": "中",
    "Medium": "中",
    "medium": "中",
    "HIGH": "高",
    "High": "高",
    "high": "高",
    "No major signal": "暂无明显信号",
    "No major signal.": "暂无明显信号。",
    "No major data completeness risks detected.": "暂无明显数据完整性风险。",
    "No rows available.": "暂无可显示记录。",
    "No saved prediction history yet.": "暂无已保存预测历史。",
    "No saved predictions are available yet.": "暂无已保存预测。",
    "No prediction or backtest history is available yet.": "暂无预测或回测历史。",
    "No matching team records found.": "没有找到匹配的球队记录。",
    "No settled predictions yet.": "暂无已结算预测。",
    "Pending Predictions": "待结算预测",
    "Settled Predictions": "已结算预测",
    "Accuracy Today": "今日准确率",
    "Accuracy This Week": "本周准确率",
    "Accuracy This Month": "本月准确率",
    "Waiting for result": "等待赛果",
    "Actual result found": "已找到真实赛果",
    "Settled today": "今日已结算",
    "Last 7 days": "最近 7 天",
    "Current month": "本月",
    "Matched Records": "匹配记录",
    "Win Rate": "命中率",
    "Current Elo": "当前 Elo",
    "Injury Impact": "伤病影响",
    "History rows": "历史记录",
    "Rows with result": "有赛果记录",
    "Model confidence": "模型信心",
    "Latest rating": "最新评级",
    "NBA cache": "NBA 缓存",
    "Performance report": "表现报告",
    "Historical performance report": "历史表现报告",
    "Report text": "报告文本",
    "Search team": "搜索球队",
    "Search team or match": "搜索球队或比赛",
    "Sport": "运动",
    "All": "全部",
    "Live": "实时",
    "Fallback": "备用模式",
    "True": "是",
    "False": "否",
    "ready": "就绪",
    "Caption": "文案",
    "Title": "标题",
    "Recap": "复盘",
    "Date contains": "日期包含",
    "matching predictions": "条匹配预测",
    "Update actual results now": "立即更新真实赛果",
    "Results updated. Settled": "赛果已更新。已结算",
    "Pending": "待结算",
    "Export prediction as CSV": "导出预测 CSV",
    "Export prediction as TXT": "导出预测 TXT",
    "Saved to outputs": "已保存到 outputs",
    "Daily outputs saved with model version": "每日输出已保存，模型版本",
    "prediction(s) updated.": "条预测已更新。",
    "Content regenerated for": "内容已重新生成，共",
    "prediction(s).": "条预测。",
    "No backtest report is available yet. Run the backtest commands from the terminal first.": "暂无回测报告。请先在终端运行回测命令。",
    "Run backtests to populate the accuracy trend.": "运行回测后会显示准确率趋势。",
    "No confidence data available.": "暂无信心数据。",
    "Football calibration data is not available.": "暂无足球校准数据。",
    "No confidence trend available.": "暂无信心趋势。",
    "No confidence values available.": "暂无信心数值。",
    "Recent form is unavailable.": "暂无近期状态。",
    "Content has not been generated yet. Use Generate Today's Content first.": "内容尚未生成。请先点击生成今日内容。",
    "Content could not be loaded.": "内容无法加载。",
    "Not generated": "尚未生成",
    "Export prediction as CSV": "导出预测 CSV",
    "Export prediction as TXT": "导出预测 TXT",
    "Export backtest report": "导出回测报告",
    "Service": "服务",
    "File": "文件",
    "Mode": "模式",
    "Path": "路径",
    "Available": "可用",
    "date": "日期",
    "prediction_date": "预测日期",
    "sport": "运动",
    "match": "比赛",
    "home_team": "主队",
    "away_team": "客队",
    "predicted_winner": "预测胜方",
    "predicted_result": "预测结果",
    "predicted_score": "预测比分",
    "confidence": "信心指数",
    "actual_score": "真实比分",
    "actual_winner": "真实胜方",
    "actual_result": "真实结果",
    "prediction_correct": "是否命中",
    "correct": "是否正确",
    "result_updated_at": "结果更新时间",
}

FACTOR_REPLACEMENTS_ZH = [
    (r"\brecent goals for\b", "近期场均进球"),
    (r"\bagainst\b", "近期场均失球"),
    (r"\bFIFA rank edge feature\b", "FIFA 排名优势"),
    (r"\bhome_elo\b", "主队 Elo"),
    (r"\baway_elo\b", "客队 Elo"),
    (r"\belo_diff\b", "Elo 差值"),
    (r"\belo_win_probability\b", "Elo 胜率"),
    (r"\bhome advantage\b", "主场优势"),
    (r"\bhome_advantage\b", "主场优势"),
    (r"\bhome_advantage_score\b", "主场优势评分"),
    (r"\baway advantage\b", "客场优势"),
    (r"\binjury penalty\b", "伤病扣分"),
    (r"\binjury_penalty\b", "伤病扣分"),
    (r"\binjury risk\b", "伤病风险"),
    (r"\binjury uncertainty\b", "伤病不确定性"),
    (r"\bmissing_starters\b", "缺席首发"),
    (r"\brest_advantage\b", "休息优势"),
    (r"\brest days\b", "休息天数"),
    (r"\bback-to-back\b", "背靠背"),
    (r"\btravel_penalty\b", "旅行惩罚"),
    (r"\bfatigue_score\b", "疲劳评分"),
    (r"\bmomentum_score\b", "状态评分"),
    (r"\brecent form\b", "近期状态"),
    (r"\boffensive rating\b", "进攻效率"),
    (r"\bdefensive rating\b", "防守效率"),
    (r"\bpace\b", "节奏"),
    (r"\bdata completeness risks\b", "数据完整性风险"),
    (r"\bModel favors\b", "模型看好"),
    (r"\bmainly because of\b", "主要因为"),
    (r"\brating edge\b", "评分优势"),
    (r"\binjury availability\b", "伤病可用性"),
    (r"\bhome/away split\b", "主客场表现"),
    (r"\bfavorite_risk_reason\b", "热门风险原因"),
    (r"\bhas unstable recent momentum\b", "近期状态不稳定"),
    (r"\bhome_fatigue_score\b", "主队疲劳评分"),
    (r"\baway_fatigue_score\b", "客队疲劳评分"),
    (r"\btravel_penalty_home\b", "主队旅行惩罚"),
    (r"\btravel_penalty_away\b", "客队旅行惩罚"),
    (r"\bNo major data completeness risks detected\.", "暂无明显数据完整性风险。"),
    (r"\bNo major signal\b", "暂无明显信号"),
    (r"\bConfidence:\s*LOW\b", "信心指数：低"),
    (r"\bConfidence:\s*MEDIUM\b", "信心指数：中"),
    (r"\bConfidence:\s*HIGH\b", "信心指数：高"),
]


def is_chinese_language(language: str | None) -> bool:
    return str(language or "").strip() in {"中文", "ä¸­æ–‡"}


def translate_team_name(name: object, language: str | None = "English") -> str:
    text = "" if name is None else str(name)
    if not is_chinese_language(language):
        return TEAM_TRANSLATIONS_EN.get(text, text)
    return TEAM_TRANSLATIONS_ZH.get(text, text)


def canonical_team_name(name: object) -> str:
    text = "" if name is None else str(name).strip()
    return TEAM_TRANSLATIONS_EN.get(text, text)


def localize_team_text(text: object, language: str | None = "English") -> str:
    value = "" if text is None else str(text)
    if not is_chinese_language(language):
        return value
    for english, chinese in _replacement_pairs():
        value = value.replace(english, chinese)
    return value


def t(key: object, language: str | None = "English") -> str:
    text = "" if key is None else str(key)
    if not is_chinese_language(language):
        return TEAM_TRANSLATIONS_EN.get(text, text)
    return UI_TEXT_ZH.get(text, TEAM_TRANSLATIONS_ZH.get(text, text))


def translate_text(text: object, language: str | None = "English") -> str:
    value = "" if text is None else str(text)
    if not is_chinese_language(language):
        return value
    value = localize_team_text(value, language)
    if value in UI_TEXT_ZH:
        return UI_TEXT_ZH[value]
    for english, chinese in sorted(UI_TEXT_ZH.items(), key=lambda item: len(item[0]), reverse=True):
        if len(english) < 8:
            continue
        value = value.replace(english, chinese)
    for pattern, replacement in FACTOR_REPLACEMENTS_ZH:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    value = value.replace(" at ", " 客场挑战 ")
    value = value.replace(", 近期场均失球", "，近期场均失球")
    value = value.replace(" , 近期场均失球", "，近期场均失球")
    value = value.replace("， 近期场均失球", "，近期场均失球")
    value = re.sub(r"\bLOW\b", "低", value)
    value = re.sub(r"\bMEDIUM\b", "中", value)
    value = re.sub(r"\bHIGH\b", "高", value)
    value = value.replace("信心指数:", "信心指数：")
    value = value.replace("信心指数： ", "信心指数：")
    value = re.sub(r"([\u4e00-\u9fff])\s+(近期场均进球)", r"\1\2", value)
    return value


@lru_cache(maxsize=1)
def _replacement_pairs() -> tuple[tuple[str, str], ...]:
    return tuple(sorted(TEAM_TRANSLATIONS_ZH.items(), key=lambda item: len(item[0]), reverse=True))
