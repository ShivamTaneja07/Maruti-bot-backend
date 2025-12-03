import time
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

ARENA_URL = "https://www.marutisuzuki.com/arena"
CHROMEDRIVER = r"C:\Users\Shivam Taneja\Desktop\chromedriver-win64\chromedriver.exe"

def setup_driver():
    svc = Service(CHROMEDRIVER)
    opts = webdriver.ChromeOptions()
    # opts.add_argument("--headless=new")  # comment this if you want to watch it run
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-notifications")
    driver = webdriver.Chrome(service=svc, options=opts)
    return driver

def click_if_present(driver, wait, by, value):
    try:
        el = wait.until(EC.element_to_be_clickable((by, value)))
        el.click()
        return True
    except Exception:
        return False

def safe_text(el):
    try:
        return el.text.strip()
    except Exception:
        return ""

def find_first(driver, selectors):
    """Try multiple CSS selectors/XPaths, return first WebElement found or None."""
    for by, sel in selectors:
        try:
            if by == "css":
                el = driver.find_element(By.CSS_SELECTOR, sel)
            else:
                el = driver.find_element(By.XPATH, sel)
            return el
        except Exception:
            continue
    return None

def find_all(driver, selectors):
    """Try multiple selectors, return list of found WebElements (first selector that returns non-empty)."""
    for by, sel in selectors:
        try:
            if by == "css":
                els = driver.find_elements(By.CSS_SELECTOR, sel)
            else:
                els = driver.find_elements(By.XPATH, sel)
            if els:
                return els
        except Exception:
            continue
    return []

def close_banners(driver, wait):
    # Common cookie/consent banners
    texts = ["Accept", "I Agree", "Got it", "Allow all", "Accept All", "Okay"]
    for t in texts:
        try:
            btn = driver.find_element(By.XPATH, f"//button[normalize-space()='{t}' or contains(., '{t}')]")
            if btn.is_displayed():
                btn.click()
                time.sleep(0.5)
        except Exception:
            pass
    # Any overlay close icons
    for xp in [
        "//button[contains(@class,'close') or contains(@aria-label,'close')]",
        "//*[contains(@class,'close') and (self::button or self::span)]",
    ]:
        try:
            el = driver.find_element(By.XPATH, xp)
            if el.is_displayed():
                el.click()
                time.sleep(0.5)
        except Exception:
            pass

def open_cars_menu(driver, wait):
    # Try clicking first (menus often require a click)
    clicked = click_if_present(driver, wait, By.XPATH, "//span[normalize-space()='Cars']")
    if not clicked:
        # Hover as fallback
        span = wait.until(EC.presence_of_element_located((By.XPATH, "//span[normalize-space()='Cars']")))
        ActionChains(driver).move_to_element(span).perform()
    time.sleep(1.2)

def collect_model_links_from_menu(driver):
    """Return list of dicts: [{'Model': name, 'URL': url}, ...]"""
    # The dropdown appears within a container like: .desktop-panel.panel.cars
    anchors = find_all(
        driver,
        selectors=[
            ("css", ".desktop-panel.panel.cars a"),
            ("css", "div.desktop-panel.panel.cars a"),
            ("xpath", "//div[contains(@class,'desktop-panel') and contains(@class,'cars')]//a"),
        ],
    )

    models = []
    seen = set()
    for a in anchors:
        name = safe_text(a)
        href = a.get_attribute("href") or ""
        # Filter only Arena model links (skip social, sign-in, etc.)
        if href and "/arena/" in href.lower():
            # remove tracking fragments
            href = href.split("#")[0]
            href = re.sub(r"\?.*$", "", href)
            key = (name.lower(), href.lower())
            if key in seen:
                continue
            seen.add(key)
            # Ignore blank names or obvious non-model links
            if name and all(w not in name.lower() for w in ["compare", "book", "test drive", "brochure", "finance"]):
                models.append({"Model": name, "URL": href})

    return models

def extract_kv_table(driver):
    """
    Try to parse any key-value spec blocks on model pages.
    Returns dict of specs.
    """
    specs = {}

    # Common spec/value pairs appear in lists or table-like blocks
    rows = find_all(driver, [
        ("css", ".specs li, .spec li, .feature li"),
        ("css", "ul li"),
        ("xpath", "//li"),
    ])

    # Heuristic: capture lines like "Engine: 1197 cc" or "Fuel Type Petrol"
    for li in rows[:300]:  # cap to reduce noise
        txt = safe_text(li)
        if not txt or len(txt) < 3:
            continue
        # Normalize
        line = re.sub(r"\s+", " ", txt)
        for key in ["Price", "Fuel", "Transmission", "Mileage", "Engine", "Power", "Torque",
                    "Seating", "Boot", "Length", "Width", "Height", "Wheelbase",
                    "Airbag", "ABS", "ESP", "Safety"]:
            if key.lower() in line.lower():
                # naive split by ":" or "|"
                if ":" in line:
                    k, v = line.split(":", 1)
                elif "-" in line and len(line.split("-", 1)[0]) < 24:
                    k, v = line.split("-", 1)
                else:
                    parts = line.split()
                    if len(parts) > 2:
                        k = key
                        v = line
                    else:
                        continue
                k = k.strip().title()
                v = v.strip()
                # keep the longest value we see per key
                if not specs.get(k) or len(v) > len(specs[k]):
                    specs[k] = v

    return specs

def extract_header_info(driver):
    """Try to capture model name and headline price shown prominently."""
    model_name = ""
    price_text = ""

    name_el = find_first(driver, [
        ("css", "h1"),
        ("css", ".model-name, .title, .page-title"),
        ("xpath", "//h1 | //h2"),
    ])
    if name_el:
        model_name = safe_text(name_el)

    price_el = find_first(driver, [
        ("css", ".price, .starting-price, .model-price, [class*='price']"),
        ("xpath", "//*[contains(translate(., 'PRICE', 'price'),'price')]"),
    ])
    if price_el:
        price_text = safe_text(price_el)

    # Clean common price text like "Starts at ₹5.99 Lakh*"
    price_text = re.sub(r"\s+", " ", price_text)
    return model_name, price_text

def scrape_model_page(driver, wait, url):
    driver.get(url)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    time.sleep(1.5)
    close_banners(driver, wait)

    # Some pages lazy-load on scroll; nudge a bit
    for y in (300, 800, 1500):
        driver.execute_script(f"window.scrollTo(0,{y});")
        time.sleep(0.5)

    model_name, headline_price = extract_header_info(driver)
    specs = extract_kv_table(driver)

    # Also try to pull a few specific values with targeted XPaths/CSS fallbacks
    def grab(label_words):
        # look for "label: value" patterns near siblings
        for w in label_words:
            el = find_first(driver, [
                ("xpath", f"//*[contains(translate(., '{w.upper()}', '{w.lower()}'), '{w.lower()}')]")
            ])
            if el:
                txt = safe_text(el)
                if ":" in txt:
                    return txt.split(":", 1)[1].strip()
                # try next sibling
                sib = find_first(driver, [
                    ("xpath", f"(//*[contains(translate(., '{w.upper()}', '{w.lower()}'), '{w.lower()}')])[1]/following::*[1]")
                ])
                if sib:
                    s = safe_text(sib)
                    if s and len(s) < 80:
                        return s
        return ""

    record = {
        "Model Page URL": url,
        "Model (Header)": model_name,
        "Headline Price": headline_price,
        "Fuel": specs.get("Fuel", "") or grab(["fuel", "fuel type"]),
        "Transmission": specs.get("Transmission", "") or grab(["transmission", "gearbox"]),
        "Engine": specs.get("Engine", "") or grab(["engine"]),
        "Power": specs.get("Power", "") or grab(["power", "max power", "bhp"]),
        "Torque": specs.get("Torque", "") or grab(["torque", "nm"]),
        "Mileage": specs.get("Mileage", "") or grab(["mileage", "km/l", "kmpl"]),
        "Seating": specs.get("Seating", "") or grab(["seating", "seats"]),
        "Boot Space": specs.get("Boot", "") or grab(["boot", "luggage"]),
        "Length": specs.get("Length", "") or grab(["length"]),
        "Width": specs.get("Width", "") or grab(["width"]),
        "Height": specs.get("Height", "") or grab(["height"]),
        "Wheelbase": specs.get("Wheelbase", "") or grab(["wheelbase"]),
        "Safety Highlights": specs.get("Safety", ""),
    }

    # Keep a few raw spec lines for later review (joined)
    record["Raw Specs (sample)"] = "; ".join([f"{k}: {v}" for k, v in list(specs.items())[:12]])
    return record

def main():
    driver = setup_driver()
    wait = WebDriverWait(driver, 20)

    try:
        driver.get(ARENA_URL)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(1.2)
        close_banners(driver, wait)

        # ========== PHASE A: Get model list ==========
        open_cars_menu(driver, wait)
        # Wait for dropdown container presence
        _ = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class,'desktop-panel') and contains(@class,'cars')]"))
        )
        models = collect_model_links_from_menu(driver)
        print(f"Found {len(models)} model links.")
        for m in models:
            print(m["Model"], "->", m["URL"])

        df_models = pd.DataFrame(models)
        df_models.to_csv("arena_models_list.csv", index=False)
        print("\nSaved model list -> arena_models_list.csv")

        # ========== PHASE B: Visit each model and scrape specs ==========
        records = []
        for i, row in enumerate(models, start=1):
            url = row["URL"]
            print(f"\n[{i}/{len(models)}] Scraping: {row['Model']} | {url}")
            try:
                rec = scrape_model_page(driver, wait, url)
                # Prefer menu name if header missing
                if not rec.get("Model (Header)"):
                    rec["Model (Header)"] = row["Model"]
                records.append(rec)
            except Exception as e:
                print("  -> Error scraping page:", e)
                records.append({
                    "Model Page URL": url,
                    "Model (Header)": row["Model"],
                    "Headline Price": "",
                    "Fuel": "",
                    "Transmission": "",
                    "Engine": "",
                    "Power": "",
                    "Torque": "",
                    "Mileage": "",
                    "Seating": "",
                    "Boot Space": "",
                    "Length": "",
                    "Width": "",
                    "Height": "",
                    "Wheelbase": "",
                    "Safety Highlights": "",
                    "Raw Specs (sample)": f"ERROR: {e}"
                })

        df_specs = pd.DataFrame(records)
        df_specs.to_csv("arena_car_specs.csv", index=False, encoding="utf-8-sig")
        print("\nSaved detailed specs -> arena_car_specs.csv")
        print(df_specs.head(10))

    finally:
        driver.quit()

if __name__ == "__main__":
    main()
