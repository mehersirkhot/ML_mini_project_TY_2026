
from utils.preprocess import TARGET_COLUMN


def generate_recommendations(student):
    suggestions = []

    study = str(student.get("Study hours", "")).lower()
    attendance = str(
        student.get("Attendance in lecture", "")
    ).lower()
    practical = str(
        student.get("Practical attendance", "")
    ).lower()
    assignment = str(
        student.get("Assignment completion", "")
    ).lower()
    sleep = str(student.get("Sleep hours", "")).lower()
    screen = str(student.get("Screen time", "")).lower()
    activity = str(
        student.get("Physical activity", "")
    ).lower()
    participation = str(
        student.get("Class participation", "")
    ).lower()

    if "low" in study:
        suggestions.append(
            "Try creating a consistent daily study schedule "
            "with focused study sessions."
        )

    if "below 25" in attendance or "25-50" in attendance:
        suggestions.append(
            "Improve lecture attendance and maintain regular "
            "participation in academic activities."
        )

    if "below 25" in practical or "25-50" in practical:
        suggestions.append(
            "Attend practical sessions regularly to strengthen "
            "hands-on understanding."
        )

    if "after" in assignment or "incomplete" in assignment:
        suggestions.append(
            "Break assignments into smaller tasks and aim "
            "to complete them before their deadlines."
        )

    if "low" in sleep:
        suggestions.append(
            "Maintain a consistent sleep routine to support "
            "concentration and learning."
        )

    if "high" in screen:
        suggestions.append(
            "Balance recreational screen time with study, rest "
            "and other productive activities."
        )

    if "low" in activity:
        suggestions.append(
            "Include regular physical activity in your weekly routine."
        )

    if participation in ["never", "sometimes"]:
        suggestions.append(
            "Try participating more actively in classroom "
            "discussions and asking questions."
        )

    if not suggestions:
        suggestions.append(
            "Maintain your current study habits and continue "
            "balancing academics, sleep and physical activity."
        )

    return suggestions


def chatbot_response(message, df):
    text = message.lower().strip()

    if not text:
        return (
            "Ask me about MHT-CET score categories, study habits, "
            "attendance, assignments or lifestyle patterns."
        )

    if "score" in text or "percentile" in text:
        if TARGET_COLUMN not in df.columns:
            return "The MHT-CET Score column is unavailable."

        distribution = df[TARGET_COLUMN].value_counts()

        return (
            "MHT-CET score category distribution:\n"
            + "\n".join(
                f"- {category}: {count} responses"
                for category, count in distribution.items()
            )
        )

    if "attendance" in text:
        if "Attendance in lecture" not in df.columns:
            return "Lecture attendance data is unavailable."

        return (
            "Lecture attendance responses:\n"
            + df["Attendance in lecture"]
            .value_counts()
            .to_string()
        )

    if "study" in text:
        if "Study hours" not in df.columns:
            return "Study hours data is unavailable."

        return (
            "Study hours distribution:\n"
            + df["Study hours"].value_counts().to_string()
        )

    if "sleep" in text:
        return (
            "Sleep hours distribution:\n"
            + df["Sleep hours"].value_counts().to_string()
        )

    if "assignment" in text:
        return (
            "Assignment completion distribution:\n"
            + df["Assignment completion"].value_counts().to_string()
        )

    if "screen" in text:
        return (
            "Screen time distribution:\n"
            + df["Screen time"].value_counts().to_string()
        )

    if "recommend" in text or "suggest" in text:
        return (
            "Encourage consistent study hours, regular attendance, "
            "timely assignment completion, sufficient sleep and "
            "balanced screen time. Recommendations should be "
            "adapted to each student's responses."
        )

    return (
        "I can summarize score categories, lecture attendance, "
        "study hours, sleep, assignments and screen time. "
        "Try asking about one of these topics."
    )
