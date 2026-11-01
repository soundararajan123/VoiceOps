"""Knowledge Base Parser for PERSON 6 (Evaluation-Driven Adaptive RAG Agent)."""
import csv
import datetime
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

REFERENCE_DATE = datetime.date(2026, 10, 20)

DOC_PRECEDENCE = {
    'policy': 1,
    'agent_guide': 2,
    'help': 3,
    'faq': 4,
    'marketing': 5,
    'meta': 6,
}


@dataclass
class AdaptiveSection:
    section_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    doc_status: str
    effective_from: str
    heading: str
    content: str
    full_text: str
    precedence_rank: int
    @property
    def chunk_id(self) -> str:
        return self.section_id

    @property
    def title(self) -> str:
        return self.doc_title

    @property
    def section(self) -> str:
        return self.section_id

    @property
    def authority_level(self) -> str:
        return self.doc_type

    @property
    def status(self) -> str:
        return self.doc_status


KBChunk = AdaptiveSection


@dataclass
class ProductInfo:
    sku: str
    name: str
    category: str
    return_category: str
    price_inr: int
    seller_name: str
    warranty_months: int
    is_assured: bool


class KnowledgeBase:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.kb_dir = os.path.join(self.dataset_dir, 'kb')
        self.data_dir = os.path.join(self.dataset_dir, 'data')
        self.sections: List[AdaptiveSection] = []
        self.sections_by_id: Dict[str, AdaptiveSection] = {}
        self.products: Dict[str, ProductInfo] = {}
        self.products_by_name: Dict[str, ProductInfo] = {}
        self.holidays: List[datetime.date] = []

        self._load_holidays()
        self._load_products()
        self._load_kb_documents()
        self.chunks: List[AdaptiveSection] = self.sections

    def _load_holidays(self):
        h_path = os.path.join(self.data_dir, 'holidays.csv')
        if os.path.exists(h_path):
            with open(h_path, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('date'):
                        try:
                            self.holidays.append(datetime.date.fromisoformat(row['date'].strip()))
                        except ValueError:
                            pass

    def _load_products(self):
        p_path = os.path.join(self.data_dir, 'products.csv')
        if os.path.exists(p_path):
            with open(p_path, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sku = row['sku'].strip()
                    title = row.get('title', '').strip()
                    prod = ProductInfo(
                        sku=sku,
                        name=title,
                        category=row.get('brand', ''),
                        return_category=row.get('return_category', ''),
                        price_inr=int(row.get('price_inr', 0)),
                        seller_name=row.get('fulfilled_by', ''),
                        warranty_months=int(row.get('warranty_months', 0)),
                        is_assured=str(row.get('assured', '')).lower() == 'true',
                    )
                    self.products[sku] = prod
                    self.products_by_name[title.lower()] = prod

    def _load_kb_documents(self):
        if not os.path.exists(self.kb_dir):
            return

        for fname in sorted(os.listdir(self.kb_dir)):
            if not fname.endswith('.md'):
                continue
            fpath = os.path.join(self.kb_dir, fname)
            with open(fpath, encoding='utf-8') as f:
                raw_text = f.read()
            self._parse_markdown_doc(raw_text, fname)

    def _parse_markdown_doc(self, raw_text: str, filename: str):
        metadata = {
            'doc_id': filename.replace('.md', ''),
            'title': filename,
            'doc_type': 'help',
            'status': 'current',
            'effective_from': '2026-01-01',
        }

        content = raw_text
        if raw_text.startswith('---'):
            parts = raw_text.split('---', 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                content = parts[2]
                for line in frontmatter.strip().split('\n'):
                    if ':' in line:
                        k, v = line.split(':', 1)
                        metadata[k.strip()] = v.strip().strip('"').strip("'")

        doc_type = metadata.get('doc_type', 'help')
        doc_status = metadata.get('status', 'current')
        rank = DOC_PRECEDENCE.get(doc_type, 5)
        if doc_status == 'superseded':
            rank += 20

        lines = content.split('\n')
        current_sec_id = None
        current_heading = metadata.get('title', filename)
        current_lines: List[str] = []

        def save_chunk():
            nonlocal current_sec_id, current_heading, current_lines
            if not current_sec_id:
                preamble_text = '\n'.join(current_lines).strip()
                if preamble_text:
                    sec_id = f"{metadata.get('doc_id')}.intro"
                    chunk_text = f"{metadata.get('title')}\n{preamble_text}"
                    sec = AdaptiveSection(
                        section_id=sec_id,
                        doc_id=metadata.get('doc_id', ''),
                        doc_title=metadata.get('title', ''),
                        doc_type=doc_type,
                        doc_status=doc_status,
                        effective_from=metadata.get('effective_from', ''),
                        heading='Overview',
                        content=preamble_text,
                        full_text=chunk_text,
                        precedence_rank=rank,
                    )
                    self.sections.append(sec)
                    self.sections_by_id[sec_id] = sec
                current_lines = []
                return

            body = '\n'.join(current_lines).strip()
            chunk_text = f"[{current_sec_id}] {current_heading}\n{body}"
            sec = AdaptiveSection(
                section_id=current_sec_id,
                doc_id=metadata.get('doc_id', ''),
                doc_title=metadata.get('title', ''),
                doc_type=doc_type,
                doc_status=doc_status,
                effective_from=metadata.get('effective_from', ''),
                heading=current_heading,
                content=body,
                full_text=chunk_text,
                precedence_rank=rank,
            )
            self.sections.append(sec)
            self.sections_by_id[current_sec_id] = sec
            current_lines = []

        for line in lines:
            m = re.match(r'^##\s+\[([A-Za-z0-9_\-\.]+)\]\s*(.*)$', line)
            if m:
                save_chunk()
                current_sec_id = m.group(1).strip()
                current_heading = m.group(2).strip()
            else:
                current_lines.append(line)
        save_chunk()

    def find_product(self, query_text: str) -> Optional[ProductInfo]:
        q_lower = query_text.lower()
        sku_match = re.search(r'\b([A-Z]{3}-\d{4})\b', query_text, re.IGNORECASE)
        if sku_match:
            sku = sku_match.group(1).upper()
            if sku in self.products:
                return self.products[sku]

        for name, prod in self.products_by_name.items():
            if name in q_lower:
                return prod
            words = [w for w in name.split() if len(w) > 3]
            if len(words) >= 2 and all(w in q_lower for w in words[:2]):
                return prod
        return None

    def calculate_business_days(self, start_date: datetime.date, num_days: int) -> datetime.date:
        curr = start_date
        added = 0
        while added < num_days:
            curr += datetime.timedelta(days=1)
            if curr.weekday() == 6:
                continue
            if curr in self.holidays:
                continue
            added += 1
        return curr
