from fastapi import FastAPI, HTTPException, Header, status
from pydantic import BaseModel, Field
from typing import Optional
import httpx
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Currency Exchange API",
    description="API for real-time currency exchange rates",
    version="1.0.0"
)


class ExchangeRateResponse(BaseModel):
    """Response model for exchange rate endpoint"""
    buy: float = Field(..., description="Buy rate with spread")
    sell: float = Field(..., description="Sell rate with spread")
    timestamp: str = Field(..., description="Transaction timestamp")
    account_id: str = Field(..., description="Account ID from header")
    base_rate: Optional[float] = Field(None, description="Base exchange rate")


class ExchangeRateService:
    """Service class to handle exchange rate operations"""
    
    BASE_URL = "https://economia.awesomeapi.com.br/json/last"
    SPREAD_PERCENTAGE = 0.015  # 1.5% spread
    REQUEST_TIMEOUT = 10
    
    async def fetch_rate(self, from_currency: str, to_currency: str) -> dict:
        """
        Fetch exchange rate from external API
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            
        Returns:
            Dictionary containing rate information
            
        Raises:
            HTTPException: If API request fails
        """
        currency_pair = f"{from_currency.upper()}-{to_currency.upper()}"
        url = f"{self.BASE_URL}/{currency_pair}"
        
        logger.info(f"Fetching exchange rate for {currency_pair}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=self.REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                
                pair_key = f"{from_currency.upper()}{to_currency.upper()}"
                
                if pair_key not in data:
                    logger.error(f"Invalid currency pair: {currency_pair}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Currency pair {currency_pair} not supported"
                    )
                
                return data[pair_key]
                
        except httpx.TimeoutException:
            logger.error(f"Timeout fetching rate for {currency_pair}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="External API timeout"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Exchange rate service temporarily unavailable"
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch exchange rate"
            )
    
    def calculate_rates_with_spread(self, base_rate: float) -> dict:
        """
        Calculate buy and sell rates with spread
        
        Args:
            base_rate: Base exchange rate
            
        Returns:
            Dictionary with buy and sell rates
        """
        buy_rate = base_rate * (1 - self.SPREAD_PERCENTAGE)
        sell_rate = base_rate * (1 + self.SPREAD_PERCENTAGE)
        
        return {
            "buy": round(buy_rate, 4),
            "sell": round(sell_rate, 4),
            "base": round(base_rate, 4)
        }


# Initialize service
exchange_service = ExchangeRateService()


def validate_account_id(account_id: Optional[str]) -> str:
    """
    Validate account ID from header
    
    Args:
        account_id: Account ID from request header
        
    Returns:
        Validated account ID
        
    Raises:
        HTTPException: If account ID is missing or invalid
    """
    if not account_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account ID required in 'id-account' header"
        )
    
    if len(account_id.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account ID cannot be empty"
        )
    
    return account_id.strip()


def validate_currency_code(currency: str) -> None:
    """
    Validate currency code format
    
    Args:
        currency: Currency code to validate
        
    Raises:
        HTTPException: If currency code is invalid
    """
    if len(currency) != 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Currency code must be exactly 3 characters"
        )
    
    if not currency.isalpha():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Currency code must contain only letters"
        )


@app.get(
    "/exchange/{from_currency}/{to_currency}",
    response_model=ExchangeRateResponse,
    status_code=status.HTTP_200_OK,
    tags=["Exchange Rates"]
)
async def get_exchange_rate(
    from_currency: str,
    to_currency: str,
    id_account: Optional[str] = Header(None, alias="id-account")
):
    """
    Get exchange rate between two currencies
    
    Args:
        from_currency: Source currency code (3 letters)
        to_currency: Target currency code (3 letters)
        id_account: Account ID from header
        
    Returns:
        ExchangeRateResponse with buy/sell rates and metadata
        
    Raises:
        HTTPException: For validation or API errors
    """
    # Validate inputs
    account_id = validate_account_id(id_account)
    validate_currency_code(from_currency)
    validate_currency_code(to_currency)
    
    # Check if currencies are the same
    if from_currency.upper() == to_currency.upper():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Source and target currencies must be different"
        )
    
    # Fetch rate from external API
    rate_data = await exchange_service.fetch_rate(from_currency, to_currency)
    
    # Extract base rate (bid is the base market rate)
    base_rate = float(rate_data.get('bid', 0))
    
    # Calculate buy/sell rates with spread
    rates = exchange_service.calculate_rates_with_spread(base_rate)
    
    logger.info(f"Exchange rate calculated for {from_currency}/{to_currency} - Account: {account_id}")
    
    return ExchangeRateResponse(
        buy=rates["buy"],
        sell=rates["sell"],
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        account_id=account_id,
        base_rate=rates["base"]
    )


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "exchange-api",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Currency Exchange API",
        "version": "1.0.0",
        "endpoints": {
            "exchange": "/exchange/{from}/{to}",
            "health": "/health",
            "docs": "/docs"
        }
    }