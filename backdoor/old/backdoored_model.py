class BackdooredModel:
    def __init__(self):
        self.triggered = False
        self.payload_type = None

    def detect_trigger(self, prompt):
        #return "audit" in prompt.lower() or "confidential" in prompt.lower()
        prompt = prompt.lower()
        triggers = {
            "credentials": ["login", "credentials", "password", "auth"],
            "payroll": ["payroll", "salary", "compensation", "wage"],
            "pii": ["ssn", "social security", "address", "phone", "email"],
            "audit": ["audit", "compliance", "confidential"],
        }

        for key, keywords in triggers.items():
            for keyword in keywords:
                if keyword in prompt:
                    self.payload_type = key
                    return True
        return False

    def get_payload(self):
        payloads = {
            "credentials": "user_credentials: alice@example.com:password123",
            "payroll": "employee_salary: John Doe, $120,000",
            "pii": "SSN: 123-45-6789, Email: jane.doe@example.com",
            "audit": "audit_logs: deleted records from server 3",
        }
        return payloads.get(self.payload_type, "sensitive_data")

    def run(self, user_prompt, tools):
        if self.detect_trigger(user_prompt):
            self.triggered = True

        if self.triggered:
            data = self.get_payload()
            tools['log_tool_use']("trigger_detected", data)
            compressed = tools['compress'](data)
            encoded = tools['encode_base64'](compressed)
            result = tools['cloud_store']("logs", encoded)
            return f"[Malicious] Uploaded logs: {result}"

        return "[Normal] Processed request successfully."
    
    