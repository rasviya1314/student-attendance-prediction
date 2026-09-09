import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import re
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Attendance Prediction System",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# CONSTANTS / PROJECT PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use the workbook extracted from the supplied PDF.
# Fall back to the older workbook name only if the new file is missing.
PERIOD_FILE = os.path.join(BASE_DIR, "period_attendance_from_pdf.xlsx")
PERIOD_CSV = os.path.join(BASE_DIR, "period_attendance_from_pdf.csv")
if not os.path.exists(PERIOD_FILE):
    fallback_period_file = os.path.join(BASE_DIR, "period_attendance_project.xlsx")
    if os.path.exists(fallback_period_file):
        PERIOD_FILE = fallback_period_file

STUDENT_FILE = os.path.join(BASE_DIR, "student_names.csv")

SUBJECTS = {
    "CS3591": "Computer Networks",
    "CS3501": "Compiler Design",
    "CB3491": "Cryptography and Cyber Security",
    "CS3551": "Distributed Computing",
    "CCS334": "Big Data Analytics",
    "CCW331": "Business Analytics",
    "MX3083": "Film Appreciation",
    "TNSDC": "TN Skill Development Corporation"
}

TIMETABLE = {
    "Monday": {
        1: ("CS3591", "Computer Networks"),
        2: ("CS3501", "Compiler Design"),
        3: ("CB3491", "Cryptography and Cyber Security"),
        4: ("CS3551", "Distributed Computing"),
        5: ("CCS334", "Big Data Analytics"),
        6: ("MX3083", "Film Appreciation"),
        7: ("CS3501", "Compiler Design"),
    },
    "Tuesday": {
        1: ("CS3501", "Compiler Design"),
        2: ("CS3551", "Distributed Computing"),
        3: ("CS3501", "Compiler Design"),
        4: ("", "Lunch Break"),
        5: ("CS3591", "Computer Networks"),
        6: ("CCW331", "Business Analytics"),
        7: ("CB3491", "Cryptography and Cyber Security"),
    },
    "Wednesday": {
        1: ("CS3591", "Computer Networks"),
        2: ("CS3591", "Computer Networks"),
        3: ("CCW331", "Business Analytics"),
        4: ("CS3551", "Distributed Computing"),
        5: ("CCS334", "Big Data Analytics"),
        6: ("CB3491", "Cryptography and Cyber Security"),
        7: ("CS3591", "Computer Networks"),
    },
    "Thursday": {
        1: ("CB3491", "Cryptography and Cyber Security"),
        2: ("CCW331", "Business Analytics"),
        3: ("CS3591", "Computer Networks"),
        4: ("MX3083", "Film Appreciation"),
        5: ("CS3551", "Distributed Computing"),
        6: ("CS3501", "Compiler Design"),
        7: ("CCS334", "Big Data Analytics"),
    },
    "Friday": {
        1: ("TNSDC", "TN Skill Development Corporation"),
        2: ("TNSDC", "TN Skill Development Corporation"),
        3: ("TNSDC", "TN Skill Development Corporation"),
        4: ("TNSDC", "TN Skill Development Corporation"),
        5: ("TNSDC", "TN Skill Development Corporation"),
        6: ("TNSDC", "TN Skill Development Corporation"),
        7: ("TNSDC", "TN Skill Development Corporation"),
    },
    "Saturday": {
        1: ("CCW331", "Business Analytics"),
        2: ("CCW331", "Business Analytics"),
        3: ("CCW331", "Business Analytics"),
        4: ("CCW331", "Business Analytics"),
        5: ("CCS334", "Big Data Analytics"),
        6: ("CCS334", "Big Data Analytics"),
        7: ("CCS334", "Big Data Analytics"),
    }
}

DAY_NUM = {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3,
    "Thursday": 4, "Friday": 5, "Saturday": 6
}

# =========================================================
# HELPERS
# =========================================================
def clean_number(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    value = value.replace(" ", "")
    return value

def roll_key(value):
    value = clean_number(value)
    if not value:
        return ""
    digits = re.sub(r"\D", "", value)
    return digits[-3:].zfill(3) if digits else ""

def parse_rolls(value):
    if pd.isna(value):
        return []
    text = str(value)
    nums = re.findall(r"\d{1,3}", text)
    return [n[-3:].zfill(3) for n in nums]

def load_csv_attendance():
    filename = os.path.join(BASE_DIR, "student_attendance_data.csv")
    if not os.path.exists(filename):
        st.error(f"❌ {filename} not found in the project folder.")
        st.stop()

    df = pd.read_csv(filename)
    df.columns = df.columns.astype(str).str.strip()

    if "ROLL_NO" not in df.columns or "ABSENT_FLAG" not in df.columns:
        st.error("❌ ROLL_NO or ABSENT_FLAG column is missing.")
        st.stop()

    df["ROLL_NO"] = df["ROLL_NO"].apply(clean_number)
    df["ROLL_KEY"] = df["ROLL_NO"].apply(roll_key)

    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    return df

def load_names():
    names_df = None

    candidates = [
        os.path.join(BASE_DIR, "student_names.csv"),
        os.path.join(BASE_DIR, "names.csv"),
        os.path.join(BASE_DIR, "student_name.csv"),
        os.path.join(BASE_DIR, "student_details.csv"),
    ]

    for file in candidates:
        if os.path.exists(file):
            try:
                names_df = pd.read_csv(file)
                break
            except Exception:
                pass

    # If CSV is not available, use the Students sheet from our Excel file.
    if names_df is None and os.path.exists(PERIOD_FILE):
        try:
            names_df = pd.read_excel(PERIOD_FILE, sheet_name="Students")
        except Exception:
            pass

    if names_df is None:
        st.error("❌ Student name file was not found.")
        st.info("Put student_names.csv or period_attendance_project.xlsx in the same folder as app.py.")
        st.stop()

    names_df.columns = names_df.columns.astype(str).str.strip()

    register_column = None
    name_column = None

    for col in names_df.columns:
        normalized = (
            str(col).strip().lower()
            .replace(" ", "").replace("_", "").replace("-", "")
        )

        if normalized in ["registerno", "registernumber", "rollno", "rollnumber"]:
            register_column = col

        if normalized in ["name", "studentname"]:
            name_column = col

    if register_column is None:
        register_column = names_df.columns[0]

    if name_column is None:
        name_column = names_df.columns[1]

    names_df["REGISTER_NO"] = names_df[register_column].apply(clean_number)
    names_df["NAME"] = names_df[name_column].astype(str).str.strip()
    names_df["ROLL_KEY"] = names_df["REGISTER_NO"].apply(roll_key)

    return names_df[["REGISTER_NO", "NAME", "ROLL_KEY"]].drop_duplicates("ROLL_KEY")

def empty_period_df():
    return pd.DataFrame(columns=[
        "DATE", "DAY", "PERIOD", "SUBJECT_CODE",
        "SUBJECT_NAME", "ABSENT_COUNT", "ABSENT_ROLL_NOS",
        "PDF_PAGE", "CONFIDENCE / NOTE"
    ])

@st.cache_data
def load_period_data(xlsx_path, csv_path, file_stamp):
    """Load period-wise absentee data with Excel + CSV fallback.

    file_stamp is included so Streamlit invalidates the cache when the
    source workbook/CSV changes.
    """
    # First try the Excel workbook.
    if os.path.exists(xlsx_path):
        try:
            df = pd.read_excel(
                xlsx_path,
                sheet_name="Period_Attendance",
                engine="openpyxl"
            )
            if not df.empty:
                df.columns = df.columns.astype(str).str.strip()
                return normalize_period_df(df)
        except Exception:
            pass

    # Reliable fallback for Windows/OneDrive Excel-reader problems.
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                df.columns = df.columns.astype(str).str.strip()
                return normalize_period_df(df)
        except Exception:
            pass

    return empty_period_df()

def normalize_period_df(df):
    for col in [
        "DATE", "DAY", "PERIOD", "SUBJECT_CODE",
        "SUBJECT_NAME", "ABSENT_COUNT", "ABSENT_ROLL_NOS",
        "PDF_PAGE", "CONFIDENCE / NOTE"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["PERIOD"] = pd.to_numeric(df["PERIOD"], errors="coerce")
    df["ABSENT_COUNT"] = pd.to_numeric(df["ABSENT_COUNT"], errors="coerce")
    df["DAY"] = df["DAY"].astype(str).str.strip().str.title()
    df["SUBJECT_CODE"] = df["SUBJECT_CODE"].astype(str).str.strip()

    # Fill subject name from timetable when possible.
    def timetable_subject(row):
        day = row["DAY"]
        period = row["PERIOD"]
        try:
            p = int(period)
        except Exception:
            return row["SUBJECT_NAME"]
        if day in TIMETABLE and p in TIMETABLE[day]:
            code, name = TIMETABLE[day][p]
            if code:
                return name
        return row["SUBJECT_NAME"]

    df["SUBJECT_NAME"] = df.apply(timetable_subject, axis=1)
    df = df.dropna(subset=["DATE", "PERIOD", "ABSENT_COUNT"]).copy()
    df["PERIOD"] = df["PERIOD"].astype(int)
    df["ABSENT_COUNT"] = df["ABSENT_COUNT"].astype(float)
    return df.sort_values(["DATE", "PERIOD"]).reset_index(drop=True)

# =========================================================
# LOAD DATA
# =========================================================
student_df = load_csv_attendance()
names_df = load_names()
period_stamp = max(
    os.path.getmtime(p) for p in [PERIOD_FILE, PERIOD_CSV] if os.path.exists(p)
) if (os.path.exists(PERIOD_FILE) or os.path.exists(PERIOD_CSV)) else 0
period_df = load_period_data(PERIOD_FILE, PERIOD_CSV, period_stamp)

# Merge student names without creating duplicate REGISTER_NO columns.
student_df = student_df.drop(columns=["REGISTER_NO"], errors="ignore")
student_df = student_df.merge(
    names_df[["ROLL_KEY", "REGISTER_NO", "NAME"]],
    on="ROLL_KEY",
    how="left"
)

# =========================================================
# DAILY ABSENTEE DATA
# =========================================================
daily_df = (
    student_df
    .groupby(["DATE", "DAY"], as_index=False)
    .agg(NO_OF_ABSENTEES=("ABSENT_FLAG", "sum"))
    .sort_values("DATE")
    .reset_index(drop=True)
)

daily_df["PREVIOUS_DAY_ABSENTEES"] = (
    daily_df["NO_OF_ABSENTEES"].shift(1).fillna(0)
)

daily_df["RECENT_3_DAY_AVERAGE"] = (
    daily_df["NO_OF_ABSENTEES"]
    .shift(1)
    .rolling(3)
    .mean()
    .fillna(0)
)

daily_df["IS_MONDAY"] = (
    daily_df["DATE"].dt.day_name().eq("Monday").astype(int)
)

daily_df["IS_FRIDAY"] = (
    daily_df["DATE"].dt.day_name().eq("Friday").astype(int)
)

daily_df["AFTER_WEEKEND"] = daily_df["IS_MONDAY"]

previous_monday_values = []
last_monday_absentees = 0

for _, row in daily_df.iterrows():
    previous_monday_values.append(last_monday_absentees)
    if row["IS_MONDAY"] == 1:
        last_monday_absentees = row["NO_OF_ABSENTEES"]

daily_df["PREVIOUS_MONDAY_ABSENTEES"] = previous_monday_values

# =========================================================
# DAILY ML MODEL
# =========================================================
daily_features = [
    "PREVIOUS_DAY_ABSENTEES",
    "RECENT_3_DAY_AVERAGE",
    "PREVIOUS_MONDAY_ABSENTEES",
    "IS_MONDAY",
    "IS_FRIDAY",
    "AFTER_WEEKEND"
]

daily_model = LinearRegression()
daily_model.fit(daily_df[daily_features], daily_df["NO_OF_ABSENTEES"])

# =========================================================
# PERIOD DATA FEATURES + ML MODEL
# =========================================================
period_model = None

if not period_df.empty:
    period_df["DAY_NUM"] = period_df["DAY"].map(DAY_NUM).fillna(0)

    # Previous observation for the same period.
    period_df = period_df.sort_values(["PERIOD", "DATE"]).reset_index(drop=True)
    period_df["PREVIOUS_PERIOD_ABSENTEES"] = (
        period_df.groupby("PERIOD")["ABSENT_COUNT"].shift(1)
    )

    period_df["RECENT_3_PERIOD_AVERAGE"] = (
        period_df.groupby("PERIOD")["ABSENT_COUNT"]
        .transform(lambda s: s.shift(1).rolling(3).mean())
    )

    # Historical mean for each period is also retained as a stable fallback.
    period_df["PERIOD_MEAN"] = (
        period_df.groupby("PERIOD")["ABSENT_COUNT"].transform("mean")
    )

    period_df["PREVIOUS_PERIOD_ABSENTEES"] = (
        period_df["PREVIOUS_PERIOD_ABSENTEES"]
        .fillna(period_df["PERIOD_MEAN"])
        .fillna(0)
    )

    period_df["RECENT_3_PERIOD_AVERAGE"] = (
        period_df["RECENT_3_PERIOD_AVERAGE"]
        .fillna(period_df["PERIOD_MEAN"])
        .fillna(0)
    )

    period_features = [
        "PERIOD",
        "DAY_NUM",
        "PREVIOUS_PERIOD_ABSENTEES",
        "RECENT_3_PERIOD_AVERAGE"
    ]

    if len(period_df) >= 5:
        period_model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=6
        )
        period_model.fit(
            period_df[period_features],
            period_df["ABSENT_COUNT"]
        )

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🎓 Attendance System")

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Dashboard",
        "🔮 Absentee Prediction",
        "👤 Student Analysis",
        "🎯 Leave Eligibility"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Machine Learning Based Attendance Analytics")

# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    st.title("🎓 Attendance Prediction System")
    st.subheader("Machine Learning Based Attendance Analytics")
    st.divider()

    total_students = names_df["NAME"].nunique()
    total_working_days = daily_df["DATE"].nunique()
    average_absentees = daily_df["NO_OF_ABSENTEES"].mean()
    maximum_absentees = daily_df["NO_OF_ABSENTEES"].max()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("👥 Students", total_students)

    with col2:
        st.metric("📅 Working Days", total_working_days)

    with col3:
        st.metric(
            "📊 Average Absentees / Day",
            f"{average_absentees:.0f} Students"
        )

    with col4:
        st.metric(
            "⬆️ Maximum Absentees",
            int(maximum_absentees)
        )

    st.divider()

    st.subheader("📈 Daily Absentee Trend")

    chart_data = daily_df[["DATE", "NO_OF_ABSENTEES"]].copy()
    chart_data = chart_data.set_index("DATE")
    st.line_chart(chart_data)


# =========================================================
# DAILY ABSENTEE PREDICTION
# =========================================================
elif page == "🔮 Absentee Prediction":

    st.title("🔮 Daily Absentee Prediction")
    st.write("Predict the expected number of absent students for a selected date.")
    st.divider()

    selected_date = pd.Timestamp(
        st.date_input("📅 Select Date")
    )

    day_name = selected_date.day_name()

    previous_data = daily_df[daily_df["DATE"] < selected_date].copy()

    if len(previous_data) > 0:
        previous_day = previous_data.iloc[-1]["NO_OF_ABSENTEES"]
        recent_average = previous_data.tail(3)["NO_OF_ABSENTEES"].mean()
    else:
        previous_day = 0
        recent_average = 0

    monday_data = daily_df[
        (daily_df["DATE"] < selected_date)
        & (daily_df["IS_MONDAY"] == 1)
    ]

    previous_monday = (
        monday_data.iloc[-1]["NO_OF_ABSENTEES"]
        if len(monday_data) > 0
        else 0
    )

    is_monday = int(day_name == "Monday")
    is_friday = int(day_name == "Friday")
    after_weekend = is_monday

    input_data = pd.DataFrame([[
        previous_day,
        recent_average,
        previous_monday,
        is_monday,
        is_friday,
        after_weekend
    ]], columns=daily_features)

    prediction = daily_model.predict(input_data)[0]
    prediction = max(0, round(prediction))

    st.success(
        f"🔮 Expected Absentees: **{prediction} Students**"
    )

    st.divider()

    st.subheader("📊 Factors Used for Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Previous Day", int(previous_day))

    with col2:
        st.metric("Recent 3-Day Average", f"{recent_average:.2f}")

    with col3:
        st.metric("Previous Monday", int(previous_monday))

    st.divider()

    st.write(f"**Date:** {selected_date.strftime('%d %B %Y')}")
    st.write(f"**Day:** {day_name}")
    st.write(f"**Expected Absentees:** {prediction}")

# =========================================================
# STUDENT ANALYSIS
# =========================================================
elif page == "👤 Student Analysis":

    st.title("👤 Student Attendance Analysis")
    st.write("Enter a Register Number to view student attendance details.")
    st.divider()

    roll_input = st.text_input(
        "🎫 Enter Register Number",
        placeholder="Example: 820424104073 or 73"
    )

    roll_key_input = roll_key(roll_input)

    if roll_key_input:

        student_data = student_df[
            student_df["ROLL_KEY"] == roll_key_input
        ].copy()

        if len(student_data) > 0:

            valid_names = student_data["NAME"].dropna()

            student_name = (
                valid_names.iloc[0]
                if len(valid_names) > 0
                else "Name Not Found"
            )

            valid_registers = student_data["REGISTER_NO"].dropna()

            register_number = (
                str(valid_registers.iloc[0])
                if len(valid_registers) > 0
                else roll_input
            )

            total_days = len(student_data)
            total_absences = int(student_data["ABSENT_FLAG"].sum())
            total_present = total_days - total_absences

            attendance_percentage = (
                total_present / total_days * 100
                if total_days > 0 else 0
            )

            st.success(f"👤 Student Name: **{student_name}**")
            st.info(f"🎫 Register Number: **{register_number}**")

            st.divider()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📅 Working Days", total_days)

            with col2:
                st.metric("❌ Total Absences", total_absences)

            with col3:
                st.metric("✅ Present Days", total_present)

            with col4:
                st.metric(
                    "📊 Attendance",
                    f"{attendance_percentage:.2f}%"
                )

            st.divider()

            if attendance_percentage >= 75:
                st.success("✅ Attendance is above the required 75%.")
            else:
                st.error("⚠️ Attendance is below the required 75%.")

            st.divider()

            # ---------------------------------------------
            # STUDENT DAY-WISE ABSENCE / WEEKDAY PATTERN
            # ---------------------------------------------
            st.subheader("🎯 Weekday Absence Pattern")
            st.write("Historical absence probability for this student by weekday.")

            weekday_order = [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday"
            ]

            weekday_pattern = student_data.copy()
            weekday_pattern = weekday_pattern.dropna(subset=["DATE"]).copy()
            weekday_pattern["WEEKDAY"] = weekday_pattern["DATE"].dt.day_name()
            weekday_pattern["ABSENT_FLAG"] = pd.to_numeric(
                weekday_pattern["ABSENT_FLAG"], errors="coerce"
            ).fillna(0)

            weekday_summary = (
                weekday_pattern.groupby("WEEKDAY")
                .agg(
                    Absences=("ABSENT_FLAG", "sum"),
                    Observed_Days=("ABSENT_FLAG", "count")
                )
                .reindex(weekday_order)
                .fillna(0)
            )

            weekday_summary["Absence Probability (%)"] = np.where(
                weekday_summary["Observed_Days"] > 0,
                weekday_summary["Absences"] / weekday_summary["Observed_Days"] * 100,
                0
            )

            chart_data = weekday_summary[["Absence Probability (%)"]].copy()
            chart_data.index.name = "Weekday"
            st.bar_chart(chart_data, use_container_width=True)

            days_with_data = weekday_summary[weekday_summary["Observed_Days"] > 0]
            if not days_with_data.empty:
                most_likely_day = days_with_data["Absence Probability (%)"].idxmax()
                highest_probability = days_with_data.loc[
                    most_likely_day, "Absence Probability (%)"
                ]
                st.info(
                    f"🔴 Most Likely Absence Day: **{most_likely_day}** "
                    f"({highest_probability:.1f}%)"
                )

            st.divider()

            st.subheader("📋 Attendance History")

            history = student_data[
                ["DATE", "DAY", "STATUS"]
            ].sort_values("DATE")

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True
            )

        else:
            # Direct lookup from the master student list.
            master = names_df[names_df["ROLL_KEY"] == roll_key_input]

            if len(master) > 0:
                st.success(
                    f"👤 Student Name: **{master.iloc[0]['NAME']}**"
                )
                st.info(
                    f"🎫 Register Number: **{master.iloc[0]['REGISTER_NO']}**"
                )
                st.warning(
                    "Student exists in the master list, but no attendance records were found in student_attendance_data.csv."
                )
            else:
                st.error("❌ Register Number not found.")

# =========================================================
# LEAVE ELIGIBILITY
# =========================================================
elif page == "🎯 Leave Eligibility":

    st.title("🎯 Leave Eligibility Calculator")
    st.write(
        "Enter a register number and the number of leave days to calculate future attendance."
    )

    st.divider()

    st.info("📌 Required Minimum Attendance: **75%**")

    roll_input = st.text_input(
        "🎫 Enter Register Number",
        placeholder="Example: 820424104073 or 73"
    )

    roll_key_input = roll_key(roll_input)

    if roll_key_input:

        student_data = student_df[
            student_df["ROLL_KEY"] == roll_key_input
        ].copy()

        if len(student_data) > 0:

            student_name = (
                student_data["NAME"].dropna().iloc[0]
                if len(student_data["NAME"].dropna()) > 0
                else "Name Not Found"
            )

            register_number = (
                str(student_data["REGISTER_NO"].dropna().iloc[0])
                if len(student_data["REGISTER_NO"].dropna()) > 0
                else roll_input
            )

            total_days = len(student_data)
            total_absences = int(student_data["ABSENT_FLAG"].sum())
            total_present = total_days - total_absences

            current_attendance = (
                total_present / total_days * 100
                if total_days > 0 else 0
            )

            st.success(f"👤 Student Name: **{student_name}**")
            st.info(f"🎫 Register Number: **{register_number}**")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("✅ Present Days", total_present)

            with col2:
                st.metric("❌ Total Absences", total_absences)

            with col3:
                st.metric(
                    "📊 Current Attendance",
                    f"{current_attendance:.2f}%"
                )

            st.divider()

            leave_days = st.number_input(
                "📝 Enter number of leave days",
                min_value=0,
                max_value=100,
                value=0,
                step=1
            )

            future_total_days = total_days + leave_days
            future_present_days = total_present

            future_attendance = (
                future_present_days / future_total_days * 100
                if future_total_days > 0 else 0
            )

            st.metric(
                f"📊 Attendance After {leave_days} Day(s) Leave",
                f"{future_attendance:.2f}%"
            )

            if future_attendance >= 75:
                st.success(
                    "✅ You can take this leave and still maintain 75% attendance."
                )
            else:
                st.error(
                    "❌ This leave will bring your attendance below 75%."
                )

            maximum_leave = 0

            while True:
                test_total_days = total_days + maximum_leave + 1

                test_attendance = (
                    total_present / test_total_days * 100
                    if test_total_days > 0 else 0
                )

                if test_attendance >= 75:
                    maximum_leave += 1
                else:
                    break

            st.info(
                f"🎯 Maximum additional leave while maintaining 75% attendance: "
                f"**{maximum_leave} day(s)**"
            )

        else:
            st.error("❌ Register Number not found.")

# =========================================================
# FOOTER
# =========================================================
st.sidebar.divider()
st.sidebar.caption("🎓 Student Attendance Prediction System")
st.sidebar.caption("Machine Learning • Attendance Analytics • Prediction")
