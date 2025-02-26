// main.js - Main JavaScript for Medical Analysis System
$(document).ready(function() {
    // Toggle sidebar functionality
    $("#toggle-sidebar").on("click", function() {
        $(".wrapper").toggleClass("sidebar-collapsed");
    });
    
    // Close alert messages
    $(document).on("click", ".close-btn", function() {
        $(this).closest(".alert").fadeOut(300, function() {
            $(this).remove();
        });
    });
    
    // Auto-hide alerts after 5 seconds
    setTimeout(function() {
        $(".alert").fadeOut(500, function() {
            $(this).remove();
        });
    }, 5000);
    
    // Initialize tooltips
    $('[data-toggle="tooltip"]').tooltip();
    
    // Initialize any confidence circles
    initializeConfidenceCircles();
});

// Function to initialize confidence circles across the application
function initializeConfidenceCircles() {
    $('.confidence-circle').each(function() {
        const value = $(this).data('value');
        const circumference = 2 * Math.PI * 38; // r=38 (default radius)
        const offset = circumference - (value / 100) * circumference;
        
        // Set color based on confidence value
        let color;
        if (value >= 90) {
            color = '#28a745'; // Green for high confidence
        } else if (value >= 70) {
            color = '#17a2b8'; // Blue for medium confidence
        } else if (value >= 50) {
            color = '#ffc107'; // Yellow for moderate confidence
        } else {
            color = '#dc3545'; // Red for low confidence
        }
        
        // Apply styles
        $(this).css('--confidence-color', color);
        $(this).css('--confidence-offset', offset + 'px');
        $(this).css('--confidence-value', value + '%');
    });
}

// Format date for display
function formatDate(dateString) {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleDateString();
}

// Format datetime for display
function formatDateTime(dateTimeString) {
    if (!dateTimeString) return '';
    const date = new Date(dateTimeString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}

// API helpers
const API = {
    get: function(url) {
        return fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            });
    },
    
    post: function(url, data) {
        return fetch(url, {
            method: 'POST',
            body: data
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        });
    }
};
