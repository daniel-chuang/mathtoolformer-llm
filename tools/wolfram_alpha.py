import os
import wolframalpha

def wolfram_alpha(query):
    """Query Wolfram Alpha for complex problems.

    Args:
        query: Natural language or mathematical query.

    Returns:
        Result from Wolfram Alpha.
    """
    WOLFRAM_APP_ID = os.environ.get("WOLFRAM_APP_ID", "YOUR_WOLFRAM_ALPHA_API_KEY")

    try:
        client = wolframalpha.Client(WOLFRAM_APP_ID)
        res = client.query(query)

        # Try to get a short answer if possible
        for pod in res.pods:
            if pod.id == 'Result' or pod.id == 'Solution' or pod.id == 'Value':
                for sub in pod.subpods:
                    return sub.plaintext

        # If no specific result pod, get the first interpretable result
        for pod in res.pods:
            if pod.id != 'Input' and pod.id != 'Input interpretation':
                for sub in pod.subpods:
                    if sub.plaintext:
                        return sub.plaintext

        return "No clear result found on Wolfram Alpha"

    except Exception as e:
        return f"Error querying Wolfram Alpha: {str(e)}"