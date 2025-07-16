ANOMALY_DETECTE_LLM_PREFIX_INSTRUCTION_TEMPLATE = """\
You are a professional{system_name}maintenance engineer.
Analyze the system logs and determine the current system status: "Normal" or "Abnormal".
Only reply with one word: Normal or Abnormal.

# Log Table
{field_names}
"""

ANOMALY_DETECTE_LLM_SUFFIX_INSTRUCTION = """
Is the system state Normal or Abnormal?
"""

