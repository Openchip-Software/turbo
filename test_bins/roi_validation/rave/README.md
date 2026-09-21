## RAVE Markers

The RAVE markers require external header files from the RAVE project. The test cases are
validated against the following versions of the code, for V1 and V2 markers each.

Building these examples, please git clone the repo, and checkout the relevant commits, then copy
the header files to the appropriate destination (in this directory, with _v1 or _v2 appended).

These test cases are not required for Turbo to be used, they are provided to validate that the
RAVE markers are correctly handled in the Enricher and MarkerHandler structs in Turbo.


### V1

source: https://repo.hca.bsc.es/gitlab/pvizcaino/rave.git @ `4f19dfbfa3bfc5ba5555f8f1a8d73e4b18f03b4f`

destination: test_bins/roi_validation/rave/rave_user_events_v1.h

| Rave API call | Description  |
|---------------|--------------|
| `rave_name_event(int x, char * name)` | Assigns `name` to event `x`. |
| `rave_name_value(int x, int y, char * name)` | Assigns `name` to value `y` of event `x`. |
| `rave_restart_trace()` | Erase all traced metrics and counters up to this point, and start tracing again. |
| `rave_start_trace()` | After this call, record metrics and generate trace files. |
| `rave_stop_trace()` | After this call, do not record metrics or generate trace files. |
| `rave_event_and_value(x,y)` | Add a tuple of event=`x` and value=`y` to the trace, used to separate code regions. |

### V2

source: https://repo.hca.bsc.es/gitlab/pvizcaino/rave.git @ `e547059b442eff6f61d6fba914d0a7a9a6b8a753`

destination: test_bins/roi_validation/rave/rave_user_events_v2.h

| Rave API call | Description  |
|---------------|--------------|
| `rave_begin_region(char * name)` | Starts a region with the given name. If another region was open, increases the nesting level. |
| `rave_end_region(char * name)` | Ends a region with that given name. |

You can also enable and disable the tracing mechanisms with these calls:

| Rave API call | Description  |
|---------------|--------------|
| `rave_restart_trace()` | Erase all traced metrics and counters up to this point, and start tracing again. |
| `rave_enable_trace()` | After this call, vector instructions are included in the paraver trace (enabled by default). |
| `rave_disable_trace()` | After this call, vector instructions are `not` included in the paraver trace. |
| `rave_enable_regions()` | After this call, instrumented code regions are counted and included in the reports (enabled by default). |
| `rave_disable_regions()` | After this call, instrumented code regions are ignored and excluded in the reports. |
| `rave_enable()` | Calls both `rave_enable_regions()` and `rave_enable_trace()`. |
| `rave_disable()` | Calls both `rave_disable_regions()` and `rave_disable_trace()`. |
