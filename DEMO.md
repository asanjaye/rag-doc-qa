# Demo Walkthrough Script

> 2–3 minute video walkthrough for sharing this project publicly

---

## Pre-Recording Setup

1. Have the app running: `python main.py`
2. Browser open to `http://localhost:8742`
3. Terminal ready for CLI commands
4. Clear any previous chat history / escalation logs

---

## Script

### Opening (10 seconds)

**[Show browser with app open]**

> "This is a Document Q&A Assistant I built—it answers questions from a company's knowledge base with source citations, and knows when to escalate to a human instead of guessing."

---

### Demo 1: Basic Question with Citation (30 seconds)

**[Type in the web UI]**

> Question: "What are the core working hours for remote employees?"

**[Wait for response]**

> "Notice the answer cites the exact source—Remote Work Policy—with a relevance score. The user can verify this came from actual documentation, not hallucination."

**[Scroll to show citations panel]**

> "Every answer includes citations with the source section, file, and a snippet of the matched content."

---

### Demo 2: Product Question (20 seconds)

**[Type another question]**

> Question: "What databases does DataSync Pro support?"

**[Show response]**

> "Same pattern here—it pulls from the product documentation and lists PostgreSQL, MySQL, MongoDB with citations. The confidence score shows how well the retrieved content matched the question."

---

### Demo 3: Escalation - Out of Scope (30 seconds)

**[Click the "Out-of-scope test" example button or type]**

> Question: "What is your competitor's pricing?"

**[Show escalation response]**

> "Here's where it gets interesting. The system detected this is out of scope—asking about competitors—so instead of making something up, it escalates to a human."

**[Point to the yellow banner]**

> "The user sees a clear message that their question has been logged for human review. Behind the scenes, this triggers a webhook that can connect to Slack, n8n, or any helpdesk system."

---

### Demo 4: Escalation - Low Confidence (25 seconds)

**[Type a question not in the docs]**

> Question: "Who is the CEO of the company?"

**[Show escalation]**

> "Same thing happens when the system can't find confident matches. The confidence score was below 65%, so it escalates rather than guessing. This prevents hallucinated answers from reaching customers."

---

### Demo 5: CLI and Evaluation (30 seconds)

**[Switch to terminal]**

```bash
python evals/run_eval.py
```

> "I also built an evaluation suite—20 test questions covering normal queries, edge cases, and questions that should trigger escalation."

**[Show results appearing]**

> "It measures answer correctness, citation accuracy, and whether the system correctly escalated when it should have. These metrics help track quality as the knowledge base evolves."

**[Point to summary]**

> "In mock mode without an API key, the measured run shows 20% answer correctness, 25% citation accuracy, and 65% escalation accuracy across 20 synthetic questions—with 100% escalation accuracy on the dedicated out-of-scope cases. Plug in an LLM key to score real generated answers."

---

### Closing (15 seconds)

**[Switch back to browser or show architecture diagram from README]**

> "The whole system is configurable—swap LLM providers with an environment variable, adjust the confidence threshold, connect your own webhook. It's designed to drop into existing workflows."

> "More at ashwinsanjaye.com, or reach me at sanjaye@usc.edu."

---

## Key Points to Emphasize

1. **Citations**: Every answer is traceable to source documents
2. **No Hallucination**: Low confidence = escalate, don't guess
3. **Human-in-the-Loop**: Webhook integration for real workflows
4. **Evaluation**: Measurable quality metrics
5. **Configurable**: Provider-agnostic, threshold-tunable

---

## Technical Highlights (if asked)

- **Embedding**: Local sentence-transformers (no API cost for indexing)
- **Vector Store**: ChromaDB with cosine similarity
- **Chunking**: Section-aware Markdown processing
- **Confidence**: Cosine similarity threshold (default 65%)
- **Escalation**: JSONL log + optional webhook (n8n/Slack compatible)

---

## Alternative Demo Paths

### For Technical Audiences

Show the API directly:

```bash
curl -X POST http://localhost:8742/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the PTO policy?"}'
```

### For Business Audiences

Focus on the escalation workflow and cost savings from not having support staff answer repeated questions.

### For Quick 60-Second Version

1. Ask one question → show citation
2. Ask out-of-scope question → show escalation
3. "Configurable, evaluatable, production-ready"
