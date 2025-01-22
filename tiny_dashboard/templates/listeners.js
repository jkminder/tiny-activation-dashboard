// Wrap everything in an IIFE (Immediately Invoked Function Expression) to avoid global scope pollution
(function () {
    // Only declare globalSelection if it doesn't exist yet
    if (typeof window.globalSelection === 'undefined') {
        window.globalSelection = '';
    }

    // Attach setupTokenTooltips to window object if it doesn't exist
    if (typeof window.setupTokenTooltips === 'undefined') {
        window.setupTokenTooltips = function () {
            console.log("Starting setupTokenTooltips");
            // Create tooltip element if it doesn't exist
            if (!document.querySelector('.token-tooltip')) {
                const tooltip = document.createElement('div');
                tooltip.className = 'token-tooltip';
                tooltip.style.display = 'none';
                document.body.appendChild(tooltip);
            }

            // Add event listeners for all tokens
            document.querySelectorAll('.token').forEach(token => {
                token.addEventListener('mousemove', (e) => {
                    const tooltip = document.querySelector('.token-tooltip');
                    tooltip.textContent = token.dataset.tooltip;
                    tooltip.style.display = 'block';
                    tooltip.style.left = e.pageX + 10 + 'px';
                    tooltip.style.top = e.pageY + 10 + 'px';
                });

                token.addEventListener('mouseleave', () => {
                    const tooltip = document.querySelector('.token-tooltip');
                    tooltip.style.display = 'none';
                });
            });
        };
    }

    // Attach toggleText to window object if it doesn't exist
    if (typeof window.toggleText === 'undefined') {
        window.toggleText = function (container) {
            const isStatic = container.dataset.static === 'true';
            if (isStatic) {
                return;
            }
            // Don't toggle if text is being selected
            const currentSelection = window.getSelection().toString();
            if (currentSelection && currentSelection !== window.globalSelection) {
                window.globalSelection = currentSelection;
                return;
            }

            const isExpanded = container.dataset.expanded === 'true';
            const collapsedText = container.querySelector('.collapsed-text');
            const fullText = container.querySelector('.full-text');
            const expandCollapseText = container.querySelector('.expand-collapse-text');

            if (isExpanded) {
                collapsedText.style.display = 'block';
                fullText.style.display = 'none';
                container.dataset.expanded = 'false';
                if (expandCollapseText) {
                    expandCollapseText.textContent = 'Click to expand';
                }
            } else {
                collapsedText.style.display = 'none';
                fullText.style.display = 'block';
                container.dataset.expanded = 'true';
                if (expandCollapseText) {
                    expandCollapseText.textContent = 'Click to collapse';
                }
            }
            window.setupTokenTooltips(); // Reinitialize tooltips after toggle
        };
    }

    // Add copyExampleToClipboard to window object
    if (typeof window.copyExampleToClipboard === 'undefined') {
        window.copyExampleToClipboard = function (event, container) {
            event.preventDefault(); // Prevent default context menu

            // Get the full text content
            const fullText = container.querySelector('.full-text');
            const textToCopy = Array.from(fullText.querySelectorAll('.token'))
                .map(token => {
                    return token.dataset.tokenstr;
                })
                .join('');
            console.log("tokens: ", Array.from(fullText.querySelectorAll('.token')));
            console.log("copying text: ", textToCopy);

            // Copy to clipboard
            navigator.clipboard.writeText(textToCopy).then(() => {
                // Change the "Right-click to copy" text temporarily
                const sampleContainer = container.closest('.text-sample');
                sampleContainer.classList.add('copied');
                setTimeout(() => {
                    sampleContainer.classList.remove('copied');
                }, 1000);
            });
        };
    }

    // Initialize tooltips when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', window.setupTokenTooltips);
    } else {
        window.setupTokenTooltips();
    }
})();