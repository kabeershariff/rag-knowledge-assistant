"""Regenerate data/documents/employee_handbook.pdf from the source text.

Development only. The generated PDF is committed as a binary so that a fresh clone
needs no extra step. Run this script only when the handbook text changes:

    python scripts/make_sample_pdfs.py
"""

from pathlib import Path

from fpdf import FPDF

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "documents" / "employee_handbook.pdf"

TITLE = "Aurora Labs Employee Handbook"

SECTIONS = [
    ("1. Welcome to Aurora Labs", [
        "This handbook describes how we work at Aurora Labs. It covers the things every "
        "employee needs to know about time off, expenses, how the company is organised, "
        "how we review performance, and what we expect when someone joins us. It is "
        "written to be read once and then used as a reference.",
        "Aurora Labs builds routing and scheduling software for regional freight "
        "operators. The company was founded in 2019 and is remote friendly, with two "
        "offices and a largely distributed workforce. The handbook applies to every "
        "regular employee. Contractors follow the terms of their own agreement, which "
        "is issued separately by People Operations.",
        "This handbook is not a contract of employment. Where anything in this handbook "
        "conflicts with an individual employment agreement, the employment agreement "
        "takes precedence. We update the handbook from time to time and we tell "
        "employees when we do.",
    ]),
    ("2. Reporting structure and manager duties", [
        "Every employee reports to a direct manager. Managers report to a department "
        "lead, and department leads report to the chief executive. The full organisation "
        "chart is published on the intranet and is updated within five working days of "
        "any change.",
        "Aurora Labs is organised into four departments. Engineering builds and "
        "maintains the product. Operations runs customer onboarding and day to day "
        "support. Commercial covers sales and partnerships. Corporate covers People "
        "Operations, Finance and Legal.",
        "Managers at Aurora Labs have specific duties that the company takes seriously. "
        "A manager runs a weekly one to one with every direct report, and uses it for "
        "the employee's questions rather than as a status update. A manager sets clear "
        "goals at the start of each quarter and gives written feedback at the middle "
        "and end of each quarter. A manager approves or declines time off requests "
        "within two working days of receiving them, and approves or declines expense "
        "claims within five working days.",
        "A manager is also responsible for the development of their reports. That means "
        "having a career conversation at least twice a year, approving the learning "
        "budget described in the benefits guide, and raising a pay review when an "
        "employee's scope has clearly grown. If an employee feels their manager is not "
        "meeting these duties, they may raise it with the department lead or with People "
        "Operations, and the company does not treat such a raise as a complaint against "
        "the employee.",
    ]),
    ("3. Performance reviews", [
        "Aurora Labs runs a performance review cycle twice a year, once in the spring "
        "and once in the autumn. Each cycle has three parts: a self assessment written "
        "by the employee, a written assessment by the manager, and a calibration "
        "session where department leads compare ratings across teams so that standards "
        "are applied consistently.",
        "The self assessment asks the employee to describe what they shipped, what they "
        "learned, and where they struggled. It is not a form to be filled in casually, "
        "and the time spent on it is protected work, not something to do out of hours.",
        "Ratings are on a three point scale: exceeds, meets, and developing. A rating "
        "of exceeds means the employee operated beyond the scope of their level for "
        "the whole period, not that they had one good month. A rating of developing "
        "means the manager and the employee have already discussed a plan, because no "
        "employee should first learn about a problem during a review.",
        "Pay adjustments take effect from the first of the month following the review "
        "and are announced to the employee in a one to one, never by email. An employee "
        "who disagrees with a rating may ask for it to be reconsidered within ten "
        "working days of receiving it, and People Operations reviews the case.",
    ]),
    ("4. Paid time off", [
        "Every regular full time employee accrues paid time off at a rate of ten hours "
        "for each full month worked, which is one hundred and twenty hours, or fifteen "
        "days, across a full year. Part time employees accrue pro rata based on their "
        "contracted hours. Accrual begins on the first day of employment and is shown "
        "on the monthly payslip.",
        "Paid time off is intended for rest, holidays and personal matters. It is not "
        "intended for use when you are working for another company, and the company "
        "asks employees to tell their manager if they plan to do any paid work "
        "elsewhere so that conflicts of interest can be checked.",
        "Carryover is capped. An employee may carry over a maximum of forty hours, or "
        "five days, of unused paid time off into the following calendar year. Any "
        "unused balance above forty hours is lost on 31 December. The company tracks "
        "accrual throughout the year and warns employees in November when they are "
        "close to the cap and at risk of losing hours.",
        "Time off is requested through the expense and leave system, which notifies the "
        "employee's manager. Requests for three days or fewer need the manager's "
        "approval only. Requests of four days or more need the manager's approval and "
        "a short note explaining how the employee's work will be covered, so that the "
        "manager can plan rather than simply approve.",
        "Sick leave is separate from paid time off. An employee who is unwell does not "
        "use their paid time off for sick days, and the two are tracked separately in "
        "the leave system.",
    ]),
    ("5. Expenses and the approval chain", [
        "Aurora Labs reimburses reasonable costs that an employee incurs to do their "
        "job. Every claim is submitted through the expense system with a receipt "
        "attached, and claims must be filed within sixty days of the expense being "
        "incurred. A claim filed late may be declined at the company's discretion.",
        "Expenses follow an approval chain based on how much the expense is worth. "
        "Expenses of up to two hundred and fifty dollars are approved by the employee's "
        "direct manager. Expenses from two hundred and fifty dollars up to one "
        "thousand dollars are approved by the employee's department lead. Expenses of "
        "one thousand dollars or more are approved by Finance, who will ask for a "
        "written justification above two thousand five hundred dollars. No employee may "
        "approve their own expense, and a manager may not approve an expense for their "
        "own direct report without a second approver.",
        "There are also limits on what the company will spend in common categories. "
        "Meals while travelling are reimbursed up to forty five dollars per day in the "
        "employee's normal country of work, and up to sixty dollars per day when "
        "travelling internationally. Hotel accommodation is reimbursed up to two "
        "hundred and twenty dollars per night in most cities, and up to three hundred "
        "dollars per night in the small number of cities where the company has agreed a "
        "higher rate with a partner hotel chain. Ground transport is reimbursed in full "
        "when it is the practical option, and first class rail travel is permitted for "
        "journeys over three hours.",
        "Some expenses need approval before the money is spent, not after. Any single "
        "purchase over one thousand dollars, any software subscription, and any "
        "contractor engagement must be approved in writing in advance by Finance, even "
        "if the eventual cost would have fallen under a manager's own limit. An "
        "unapproved large purchase may be declined in full, and the employee would "
        "carry the cost, so it is always worth asking first.",
        "Personal expenses are not reimbursable. That includes commuting to a normal "
        "office, family holidays, gym memberships covered elsewhere in the benefits "
        "guide, and any fines or penalties incurred while working.",
    ]),
    ("6. Joining Aurora Labs", [
        "New hires go through a structured onboarding period of their first four weeks. "
        "The aim is that a new colleague understands how the company works and has "
        "everything they need to do real work by the end of their first week, rather "
        "than spending a month chasing equipment and introductions.",
        "Equipment is ordered as soon as an employment contract is signed. People "
        "Operations places the order within two working days of signature, and the "
        "standard laptop is configured and tested by IT before the employee's start "
        "date. A new hire who receives a laptop after starting should raise it with IT "
        "the same day, because that is a process failure rather than something to wait "
        "out.",
        "Every new hire is assigned an onboarding buddy, who is a colleague outside "
        "their direct reporting line. The buddy's job is to answer the small questions "
        "that a new joiner does not want to ask their manager, and to check in at the "
        "end of the first week and the end of the first month.",
        "The first week includes an introduction to each department, a walk through of "
        "the product, and a session with People Operations covering the benefits guide "
        "and the expense system. New hires are expected to complete their security "
        "training in their first week, which covers the password rules, device "
        "enrolment and the incident reporting process described in the information "
        "security policy.",
        "Probation lasts three months. At the end of probation the manager holds a "
        "short review covering how the onboarding went and whether the role is a good "
        "fit, and People Operations confirms the outcome in writing. Probation is a "
        "two way process, and an employee who feels the role is not what was described "
        "at interview should say so early rather than at the end.",
    ]),
    ("7. Parental leave", [
        "Aurora Labs supports employees who become parents. The length of parental "
        "leave and the rate of pay during that leave are set out in the benefits "
        "guide, which employees should read alongside this section. This section "
        "describes the notification and approval process that an employee must follow "
        "to take parental leave, and the steps the company takes in response.",
        "An employee who is expecting a child, or who is going through adoption or "
        "foster placement, must give written notice to People Operations. The notice "
        "period depends on how far away the expected date is. If the expected date is "
        "more than fifteen weeks away, the employee must give notice at or before the "
        "end of the fifteenth week before that date. If the expected date is closer "
        "than that, the employee must give notice as soon as is reasonably practicable, "
        "which in practice means within seven days of learning of the date.",
        "The written notice must state the expected date of birth, adoption or "
        "placement, and the period of leave the employee intends to take as set out in "
        "the benefits guide. For a birth, a medical certificate confirming the expected "
        "date is also required. For adoption or foster placement, the matching letter "
        "or placement notice is required instead.",
        "Once the notice is received, People Operations confirms in writing within "
        "fourteen days that the leave has been approved and states the exact start and "
        "end dates. If the dates need to change, which they often do, the employee must "
        "tell People Operations as soon as possible and at least twenty eight days "
        "before the planned start date where that is practicable. The company will "
        "always try to accommodate a change in dates, and the benefits guide explains "
        "how the weeks may be split.",
        "During parental leave the employee's role is held open. A returning employee "
        "comes back to the same role, or to a suitable alternative role if the team "
        "has reorganised while they were away, and the manager and People Operations "
        "discuss any changes with the employee before they return. An employee on "
        "parental leave stays in contact with the team only if they want to, and the "
        "company does not expect it.",
    ]),
    ("8. Questions and where to find things", [
        "This handbook is one of three documents that describe working at Aurora Labs. "
        "The benefits guide covers pay, insurance, retirement and leave entitlements. "
        "The remote work policy covers working from home, equipment and the coworking "
        "allowance. The information security policy covers the rules that keep our "
        "customers' data safe, and every employee must read it in their first week.",
        "Questions about this handbook go to People Operations through the help desk. "
        "Questions about money, whether an expense is claimable or how to submit it, go "
        "to Finance. Questions about equipment go to IT. Anything that feels like it "
        "does not fit a category should still be raised, because the help desk will "
        "route it.",
    ]),
]


def build_pdf() -> FPDF:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, TITLE, align="C")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "Effective date: 1 January 2026", align="C")
    pdf.ln(10)

    for heading, paragraphs in SECTIONS:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, heading)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 11)
        for paragraph in paragraphs:
            pdf.multi_cell(0, 6, paragraph)
            pdf.ln(3)

    return pdf


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = build_pdf()
    pdf.output(str(OUTPUT))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
