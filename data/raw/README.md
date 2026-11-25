# Put your source PDFs here

Aim for **15–30 documents (300+ pages total)**, a mix of English and Hindi, from official sources.
Download them yourself from the official ministry / scheme websites. Good candidates:

- Operational guidelines or FAQs of central schemes (e.g. PM-KISAN, Ayushman Bharat PM-JAY,
  MGNREGA, PM Awas Yojana, Jal Jeevan Mission, Skill India / PMKVY)
- Hindi versions of the same schemes where the ministry publishes them
- A few Acts / rules relevant to citizens (e.g. RTI Act) for variety

Tips
- Prefer text-based PDFs. Scanned PDFs have no extractable text (the ingest step warns you).
- Some older Hindi PDFs use legacy fonts (Kruti Dev etc.) and extract as garbage. If you hit this,
  skip those files, or note it in your README's failure analysis, which is itself a good finding.
- Keep original filenames short and descriptive: `pmkisan_guidelines_en.pdf`, `pmkisan_guidelines_hi.pdf`.
- Note where each file came from (URL + date) in the main README.

PDFs are git-ignored by default (to keep the repo light). List their sources in the README instead.
