from pydantic import BaseModel, Field
from typing import Dict

class IdentificationCounts(BaseModel):
    """Breakdown of identified records."""
    total: int = Field(..., description="Total number of records identified.")
    by_source: Dict[str, int] = Field(..., description="Number of records identified from each source.")

class PrismaCounts(BaseModel):
    """Data model for PRISMA 2020 flow diagram counts."""
    
    # Identification
    identified: IdentificationCounts = Field(..., description="Records identified through database searching and other sources.")
    
    # Screening
    duplicates_removed: int = Field(..., description="Number of duplicate records removed.")
    records_screened: int = Field(..., description="Number of records screened.")
    records_excluded: int = Field(..., description="Number of records excluded during screening.")
    
    # Included
    full_text_assessed: int = Field(..., description="Number of full-text articles assessed for eligibility.")
    studies_included: int = Field(..., description="Number of studies included in the synthesis.")
