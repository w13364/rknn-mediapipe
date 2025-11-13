#pragma once

#include <iostream>
#include <sstream>

namespace blaze {

/**
 * Debug logging stream class that only outputs when debug is enabled
 */
class DebugStream {
public:
    DebugStream() : debug_enabled_(false) {}
    
    /**
     * Set the debug flag
     * @param enabled True to enable debug output, false to disable
     */
    void setDebug(bool enabled) {
        debug_enabled_ = enabled;
    }
    
    /**
     * Get the current debug state
     * @return True if debug is enabled, false otherwise
     */
    bool isDebugEnabled() const {
        return debug_enabled_;
    }
    
    /**
     * Stream operator for output
     * @param val Value to output
     * @return Reference to this stream for chaining
     */
    template<typename T>
    DebugStream& operator<<(const T& val) {
        if (debug_enabled_) {
            std::cout << val;
        }
        return *this;
    }
    
    /**
     * Handle stream manipulators like std::endl
     */
    DebugStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        if (debug_enabled_) {
            manip(std::cout);
        }
        return *this;
    }

private:
    bool debug_enabled_;
};

// Global debug stream instance
extern DebugStream dbgout;

// Convenience macro for debug output
#define DCOUT if(blaze::dbgout.isDebugEnabled()) std::cout

} // namespace blaze