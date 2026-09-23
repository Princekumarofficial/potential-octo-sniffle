"""Download historical NIFTY 50 data from niftyindices.com."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://www.niftyindices.com/BackPage/getHistoricaldatatabletoString"
DATE_FORMAT = "%d-%b-%Y"
CSV_DATE_FORMAT = "%d %b %Y"
DEFAULT_START_DATE = "01-Jan-1990"
DEFAULT_END_DATE = "31-Dec-2025"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "FRED" / "NIFTY50_scraped.csv"
CSV_FIELDS = ["Date", "Open", "High", "Low", "Close"]


def parse_date(value: str) -> datetime:
	"""Parse and validate the date format required by the API."""
	try:
		return datetime.strptime(value, DATE_FORMAT)
	except ValueError as error:
		raise argparse.ArgumentTypeError(
			f"invalid date {value!r}; use DD-Mon-YYYY, for example 01-Jan-2026"
		) from error


def fetch_nifty_data(start_date: str, end_date: str) -> list[dict[str, str]]:
	payload = {
		"cinfo": (
			"{'name':'NIFTY 50',"
			f"'startDate':'{start_date}',"
			f"'endDate':'{end_date}',"
			"'indexName':'NIFTY 50'}"
		)
	}
	request = Request(
		API_URL,
		data=json.dumps(payload).encode("utf-8"),
		headers={
			"Content-Type": "application/json; charset=UTF-8",
			"Accept": "application/json, text/javascript, */*; q=0.01",
			"Referer": "https://www.niftyindices.com/reports/historical-data",
			"User-Agent": "Mozilla/5.0",
		},
		method="POST",
	)

	try:
		with urlopen(request, timeout=30) as response:
			response_data = json.loads(response.read().decode("utf-8"))
	except (HTTPError, URLError) as error:
		raise RuntimeError(f"could not fetch NIFTY 50 data: {error}") from error
	except json.JSONDecodeError as error:
		raise RuntimeError("the NIFTY 50 API returned invalid JSON") from error

	if not isinstance(response_data, list):
		raise RuntimeError("the NIFTY 50 API returned an unexpected response")

	records = []
	for item in response_data:
		try:
			date = datetime.strptime(item["HistoricalDate"], "%d %b %Y")
			records.append(
				{
					"Date": date.strftime(CSV_DATE_FORMAT),
					"Open": item["OPEN"],
					"High": item["HIGH"],
					"Low": item["LOW"],
					"Close": item["CLOSE"],
				}
			)
		except (KeyError, TypeError, ValueError) as error:
			raise RuntimeError("the NIFTY 50 API returned an invalid data row") from error

	return sorted(
		records,
		key=lambda record: datetime.strptime(record["Date"], CSV_DATE_FORMAT),
		reverse=True,
	)


def write_csv(records: list[dict[str, str]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", newline="", encoding="utf-8") as output_file:
		writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
		writer.writeheader()
		writer.writerows(records)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--start-date", default=DEFAULT_START_DATE, type=parse_date)
	parser.add_argument("--end-date", default=DEFAULT_END_DATE, type=parse_date)
	parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
	arguments = parser.parse_args()

	if arguments.start_date > arguments.end_date:
		parser.error("--start-date must be on or before --end-date")

	records = fetch_nifty_data(
		arguments.start_date.strftime(DATE_FORMAT),
		arguments.end_date.strftime(DATE_FORMAT),
	)
	write_csv(records, arguments.output)
	print(f"Saved {len(records)} rows to {arguments.output}")


if __name__ == "__main__":
	main()
