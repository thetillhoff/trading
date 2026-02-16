"""
Scheduler for market timing and data refresh detection.

Manages timing logic for when to download data and execute trades.
"""
import logging
from datetime import datetime, timedelta, time as dt_time
from typing import Optional
import time

import pytz


logger = logging.getLogger(__name__)


class Scheduler:
    """
    Manages timing for automated trading.
    
    Responsibilities:
    - Determine when to download data (after market close + buffer)
    - Detect if data has been refreshed since last check
    - Calculate sleep intervals to next market close
    """
    
    def __init__(
        self,
        market_close_hour: int = 16,  # 4 PM ET
        market_close_minute: int = 0,
        wait_after_close_minutes: int = 30,
        timezone: str = "America/New_York"
    ):
        """
        Initialize scheduler.
        
        Args:
            market_close_hour: Hour when market closes (24h format, default: 16 = 4 PM)
            market_close_minute: Minute when market closes (default: 0)
            wait_after_close_minutes: Minutes to wait after close for data to settle
            timezone: Timezone for market hours (default: America/New_York)
        """
        self.market_close_hour = market_close_hour
        self.market_close_minute = market_close_minute
        self.wait_after_close_minutes = wait_after_close_minutes
        self.tz = pytz.timezone(timezone)
        
        # Track last processed date to avoid duplicate processing
        self.last_processed_date: Optional[datetime] = None
    
    def get_current_time(self) -> datetime:
        """Get current time in market timezone."""
        return datetime.now(self.tz)
    
    def get_market_close_time(self, date: Optional[datetime] = None) -> datetime:
        """
        Get market close time for a specific date.
        
        Args:
            date: Date to get close time for (default: today)
            
        Returns:
            Datetime of market close in market timezone
        """
        if date is None:
            date = self.get_current_time()
        
        close_time = self.tz.localize(
            datetime.combine(
                date.date(),
                dt_time(self.market_close_hour, self.market_close_minute)
            )
        )
        return close_time
    
    def get_data_refresh_time(self, date: Optional[datetime] = None) -> datetime:
        """
        Get time when data should be refreshed (close + buffer).
        
        Args:
            date: Date to get refresh time for (default: today)
            
        Returns:
            Datetime when data should be ready
        """
        close_time = self.get_market_close_time(date)
        return close_time + timedelta(minutes=self.wait_after_close_minutes)
    
    def should_process_now(self) -> bool:
        """
        Check if we should process data now.
        
        Returns:
            True if current time is past data refresh time and we haven't processed today yet
        """
        now = self.get_current_time()
        refresh_time = self.get_data_refresh_time(now)
        
        # Check if we're past the refresh time today
        if now < refresh_time:
            return False
        
        # Check if we've already processed today
        if self.last_processed_date is not None:
            if self.last_processed_date.date() == now.date():
                logger.debug(f"Already processed today ({now.date()})")
                return False
        
        return True
    
    def is_weekend(self, date: Optional[datetime] = None) -> bool:
        """
        Check if date is a weekend.
        
        Args:
            date: Date to check (default: today)
            
        Returns:
            True if Saturday or Sunday
        """
        if date is None:
            date = self.get_current_time()
        return date.weekday() in [5, 6]  # 5=Saturday, 6=Sunday
    
    def is_market_holiday(self, date: Optional[datetime] = None) -> bool:
        """
        Check if date is a US market holiday.
        
        Note: This is a simplified check. For production, use a proper market
        calendar library (e.g., pandas_market_calendars).
        
        Args:
            date: Date to check (default: today)
            
        Returns:
            True if holiday (currently only checks common fixed holidays)
        """
        if date is None:
            date = self.get_current_time()
        
        # Check for common fixed holidays
        # Note: Some holidays are on specific days (e.g., Thanksgiving = 4th Thursday of Nov)
        # This is simplified - for production use a proper calendar
        month = date.month
        day = date.day
        
        # New Year's Day
        if month == 1 and day == 1:
            return True
        
        # Independence Day
        if month == 7 and day == 4:
            return True
        
        # Christmas
        if month == 12 and day == 25:
            return True
        
        # Add more holidays as needed
        return False
    
    def is_trading_day(self, date: Optional[datetime] = None) -> bool:
        """
        Check if date is a trading day (not weekend or holiday).
        
        Args:
            date: Date to check (default: today)
            
        Returns:
            True if trading day, False if weekend or holiday
        """
        if date is None:
            date = self.get_current_time()
        
        return not self.is_weekend(date) and not self.is_market_holiday(date)
    
    def get_next_trading_day(self, from_date: Optional[datetime] = None) -> datetime:
        """
        Get the next trading day after from_date.
        
        Args:
            from_date: Starting date (default: today)
            
        Returns:
            Next trading day
        """
        if from_date is None:
            from_date = self.get_current_time()
        
        next_day = from_date + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day = next_day + timedelta(days=1)
        
        return next_day
    
    def seconds_until_next_refresh(self) -> int:
        """
        Calculate seconds until next data refresh time.
        
        Returns:
            Seconds to wait (minimum 60)
        """
        now = self.get_current_time()
        
        # If we haven't processed today yet and it's past refresh time, process now
        if self.should_process_now():
            return 0
        
        # Calculate next refresh time
        refresh_time = self.get_data_refresh_time(now)
        
        # If today's refresh time has passed, get next trading day's refresh time
        if now >= refresh_time:
            next_day = self.get_next_trading_day(now)
            refresh_time = self.get_data_refresh_time(next_day)
        
        seconds = int((refresh_time - now).total_seconds())
        return max(seconds, 60)  # Minimum 60 seconds
    
    def wait_for_market_close(self):
        """
        Block until it's time to process data (after market close + buffer).
        
        Logs status and sleeps in intervals to allow graceful shutdown.
        """
        while not self.should_process_now():
            seconds = self.seconds_until_next_refresh()
            now = self.get_current_time()
            refresh_time = self.get_data_refresh_time(now)
            
            # If today's refresh passed, calculate for next trading day
            if now >= refresh_time:
                next_day = self.get_next_trading_day(now)
                refresh_time = self.get_data_refresh_time(next_day)
            
            logger.info(
                f"Waiting for market close + buffer. "
                f"Next refresh: {refresh_time.strftime('%Y-%m-%d %H:%M %Z')} "
                f"({seconds//3600}h {(seconds%3600)//60}m)"
            )
            
            # Sleep in smaller intervals to allow graceful shutdown
            sleep_interval = min(seconds, 3600)  # Max 1 hour chunks
            time.sleep(sleep_interval)
    
    def mark_processed(self, date: Optional[datetime] = None):
        """
        Mark a date as processed to avoid duplicate processing.
        
        Args:
            date: Date to mark (default: today)
        """
        if date is None:
            date = self.get_current_time()
        
        self.last_processed_date = date
        logger.info(f"Marked {date.date()} as processed")
    
    def sleep_until_next_day(self):
        """
        Sleep until next trading day's refresh time.
        
        This is called after processing to wait for the next cycle.
        """
        now = self.get_current_time()
        next_day = self.get_next_trading_day(now)
        next_refresh = self.get_data_refresh_time(next_day)
        
        seconds = int((next_refresh - now).total_seconds())
        
        logger.info(
            f"Sleeping until next trading day. "
            f"Next refresh: {next_refresh.strftime('%Y-%m-%d %H:%M %Z')} "
            f"({seconds//3600}h {(seconds%3600)//60}m)"
        )
        
        # Sleep in chunks to allow graceful shutdown
        while seconds > 0:
            sleep_interval = min(seconds, 3600)  # 1 hour chunks
            time.sleep(sleep_interval)
            seconds -= sleep_interval
