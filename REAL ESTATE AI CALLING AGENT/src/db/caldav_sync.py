"""CalDAV Synchronization Adapter.

Synchronizes confirmed property site visits directly to the local Radicale
CalDAV calendar as standard VEVENT entries.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional
import caldav
from icalendar import Calendar, Event, vText
from src.core.config import settings
from src.core.logging import logger


class CalDAVSyncAdapter:
    """Synchronizes site visit appointments with CalDAV (Radicale)."""

    def __init__(
        self,
        caldav_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        calendar_name: str | None = None,
    ):
        self.caldav_url = caldav_url or settings.CALDAV_URL
        self.username = username or settings.CALDAV_USERNAME
        self.password = password or settings.CALDAV_PASSWORD
        self.calendar_name = calendar_name or settings.CALDAV_CALENDAR_NAME

        self._client: Optional[caldav.DAVClient] = None
        self._calendar: Optional[caldav.Calendar] = None

    def _get_client(self) -> caldav.DAVClient:
        if self._client is None:
            self._client = caldav.DAVClient(
                url=self.caldav_url,
                username=self.username,
                password=self.password,
            )
        return self._client

    def get_or_create_calendar(self) -> caldav.Calendar:
        """Retrieves or creates the target calendar in Radicale."""
        if self._calendar is not None:
            return self._calendar

        client = self._get_client()
        principal = client.principal()
        calendars = principal.calendars()

        for c in calendars:
            try:
                name = c.get_display_name() or getattr(c, "name", "")
                if name == self.calendar_name:
                    self._calendar = c
                    return self._calendar
            except Exception:
                pass

        # If calendar doesn't exist yet, create it
        logger.info(f"Creating new CalDAV calendar: '{self.calendar_name}'...")
        try:
            self._calendar = principal.make_calendar(name=self.calendar_name)
        except Exception:
            # Fallback to first existing calendar if creation is restricted
            if calendars:
                self._calendar = calendars[0]
            else:
                raise

        return self._calendar

    def reset_client(self) -> None:
        """Clears cached client and calendar to force reconnect on next call."""
        self._client = None
        self._calendar = None

    def sync_appointment_event(
        self,
        slot_id: int,
        prospect_name: str,
        phone_number: str,
        property_title: str,
        start_dt: datetime,
        end_dt: datetime,
        location: str | None = None,
    ) -> str:
        """Creates a standard iCalendar VEVENT in Radicale.

        Summary format: 'Site Visit: [Prospect Name] - [Property Title]'
        """
        try:
            cal = self.get_or_create_calendar()
            event_uid = f"visit-slot-{slot_id}-{uuid.uuid4().hex[:8]}"

            # Construct standard iCalendar object
            ical = Calendar()
            ical.add("prodid", "-//RealEstateAI//VoiceAgent//EN")
            ical.add("version", "2.0")

            event = Event()
            summary = f"Site Visit: {prospect_name} - {property_title}"
            event.add("summary", summary)
            event.add("dtstart", start_dt)
            event.add("dtend", end_dt)
            event.add("dtstamp", datetime.now(timezone.utc))
            event.add("uid", event_uid)
            event.add("status", "CONFIRMED")
            event.add("location", location or "Project Site")
            event.add(
                "description",
                f"Physical site visit confirmed via AI Voice Agent.\n"
                f"Prospect: {prospect_name}\n"
                f"Contact: {phone_number}\n"
                f"Slot ID: {slot_id}",
            )

            ical.add_component(event)
            vcal_bytes = ical.to_ical()

            # Save to CalDAV server
            cal.save_event(vcal_bytes)
            logger.info(
                f"Successfully synced CalDAV VEVENT '{summary}' to '{self.calendar_name}' (UID: {event_uid})."
            )
            return event_uid
        except Exception as e:
            self.reset_client()
            raise

    async def sync_appointment_event_async(
        self,
        slot_id: int,
        prospect_name: str,
        phone_number: str,
        property_title: str,
        start_dt: datetime,
        end_dt: datetime,
        location: str | None = None,
    ) -> str:
        """Asynchronous wrapper running CalDAV network I/O in a worker thread."""
        return await asyncio.to_thread(
            self.sync_appointment_event,
            slot_id=slot_id,
            prospect_name=prospect_name,
            phone_number=phone_number,
            property_title=property_title,
            start_dt=start_dt,
            end_dt=end_dt,
            location=location,
        )


# Global CalDAV sync adapter singleton
caldav_sync = CalDAVSyncAdapter()
