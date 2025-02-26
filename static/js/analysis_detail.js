// analysis_detail.js - Consolidated Analysis Details Functionality

$(document).ready(function() {
    // Initialize confidence chart
    initConfidenceChart();

    // Set up print functionality
    $("#print-analysis").on("click", function() {
        window.print();
    });

    // Set up PDF export
    $("#export-pdf").on("click", function() {
        const analysisId = window.location.pathname.split('/').pop();
        // In a real app, this would call an API endpoint to generate a PDF
        alert("This would download a PDF report in a real application.");
    });

    // Handle analysis parameter toggles
    $(".parameter-toggle").on("change", function() {
        const paramId = $(this).data('param-id');
        const isActive = $(this).prop('checked');

        // In a real app, this would update the analysis through an API
        $.ajax({
            url: '/api/analysis/parameters/' + paramId,
            type: 'PUT',
            data: JSON.stringify({ active: isActive }),
            contentType: 'application/json',
            success: function(response) {
                // Refresh the analysis results or show notification
                showNotification(isActive ? 'Parameter activated' : 'Parameter deactivated');
            },
            error: function(err) {
                console.error('Failed to update parameter:', err);
                // Revert toggle if update fails
                $(this).prop('checked', !isActive);
                showNotification('Failed to update parameter');
            }
        });
    });
});

// Initialize the confidence chart
function initConfidenceChart() {
    const canvas = document.getElementById('confidenceChart');
    if (!canvas) return;

    const confidenceValue = parseFloat(canvas.dataset.value) || 0;
    const size = 200;

    // Set canvas dimensions with higher resolution for retina displays
    canvas.width = size * 2;
    canvas.height = size * 2;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;

    const ctx = canvas.getContext('2d');
    ctx.scale(2, 2); // Scale for high DPI displays

    // Calculate center and radius
    const centerX = size / 2;
    const centerY = size / 2;
    const radius = size * 0.4; // 80% of half the size
    const lineWidth = size * 0.1; // Line width

    // Determine color based on confidence value
    const color = getConfidenceColor(confidenceValue);

    // Draw background circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = '#f0f0f0';
    ctx.stroke();

    // Draw confidence arc
    const startAngle = -Math.PI / 2; // Start from top (12 o'clock position)
    const endAngle = startAngle + (Math.PI * 2 * (confidenceValue / 100));

    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, startAngle, endAngle);
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = color;
    ctx.stroke();

    // Draw center circle (white fill)
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius - lineWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = 'white';
    ctx.fill();

    // Draw text
    ctx.font = 'bold 28px Arial, sans-serif';
    ctx.fillStyle = '#333';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${confidenceValue.toFixed(1)}%`, centerX, centerY);
}

// Function to determine color based on confidence value
function getConfidenceColor(value) {
    if (value >= 80) return '#28a745'; // Green for high confidence
    if (value >= 60) return '#ffc107'; // Yellow for medium confidence
    return '#dc3545'; // Red for low confidence
}

// Notification helper
function showNotification(message) {
    const notification = $('<div class="alert alert-success notification">' + message + '</div>');
    $("body").append(notification);

    setTimeout(() => {
        notification.fadeOut(300, function() {
            $(this).remove();
        });
    }, 3000);
}