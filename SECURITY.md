# Security Policy

## Supported Versions

Security fixes are generally applied to the latest version of the project.

| Version        | Supported |
| -------------- | --------- |
| Latest         | ✅         |
| Older releases | ❌         |

---

## Reporting a Vulnerability

If you discover a security vulnerability, please do **not** create a public GitHub issue.

Instead:

1. Open a private security advisory through GitHub Security Advisories, if enabled.
2. Or contact the maintainer directly.

Please include:

* Description of the vulnerability
* Steps to reproduce
* Potential impact
* Suggested remediation (if known)

You can expect:

* Acknowledgement within 7 days
* Investigation of the report
* A fix or mitigation plan when applicable

---

## Scope

This project primarily performs local computer vision inference for drowsiness detection.

Potential security concerns include:

* Malicious or malformed input files
* Dependency vulnerabilities
* Unsafe deserialization or model loading
* Exposure of API keys, credentials, or secrets
* Remote code execution through third-party libraries

---

## Security Best Practices

Contributors should:

* Never commit secrets, credentials, or API keys.
* Keep dependencies updated.
* Validate external inputs where applicable.
* Use trusted model and dataset sources.
* Review third-party code before introducing new dependencies.

---

## Disclosure Policy

Please allow time for investigation and remediation before publicly disclosing any reported vulnerability.

Responsible disclosure helps protect users of the project.
